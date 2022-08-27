import torch
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os 
import torch.nn as nn
import argparse
import torch.backends.cudnn as cudnn
import time
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.models as models

from NCEAverage import NCEAverage
from LinearAverage import LinearAverage
from NCECriterion import NCECriterion
from utils import AverageMeter
from test import NN, kNN

parser = argparse.ArgumentParser(description='PyTorch flower102 Training')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=102, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=4096, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.07, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.99, type=float,
                    metavar='M', help='momentum for non-parametric updates')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

print('==> Preparing data..')

data_dir = '/workspace/flower102/prepare_pic'
train_dir = data_dir + '/train'
test_dir = data_dir + '/valid'

transform_train = transforms.Compose([transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224),#从中心开始裁剪，留下224*224的。（随机裁剪得到的数据更多）
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率去翻转，0.5就是50%翻转，50%不翻转
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),  #转成tensor的格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差（拿人家算好的）
    ])
   
transform_test=transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #要和训练集保持一致的标准化操作
    ])


class Flower102Instance(datasets.ImageFolder):
    """Flower102Instance Dataset.
    """
    def __getitem__(self, index):
        """
        index (int): Index
	Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img,target,index

batch_size=256

trainset = Flower102Instance(os.path.join(train_dir),transform=transform_train) 
trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=2)
testset = Flower102Instance(os.path.join(test_dir),transform=transform_test) 
testloader = DataLoader(testset, shuffle=False, batch_size=batch_size, num_workers=2)
ndata = trainset.__len__()

print(ndata)
scaler = torch.cuda.amp.GradScaler()
print('==> Building model..')
#使用resnet50的网络结构
model = models.__dict__['resnet50'](128)
fc_inputs = model.fc.in_features
model.fc= nn.Linear(fc_inputs, 102)

for param in model.parameters():
    param.requires_grad = False
 
for param in model.fc.parameters():
    param.requires_grad = True


# define leminiscate
if args.nce_k > 0:
    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
else:
    lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)
    
if args.test_only or len(args.resume)>0:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.resume)
    model.load_state_dict(checkpoint['model'])
    lemniscate = checkpoint['lemniscate']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
if device == 'cuda':
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True


# define loss function
if hasattr(lemniscate, 'K'):
    criterion = NCECriterion(ndata)
else:
    criterion = nn.CrossEntropyLoss()

model.to(device)
lemniscate.to(device)
criterion.to(device)   

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(trainloader, model, criterion, optimizer, epoch, args):
    
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    # 切换到模型的训练模式
    model.train()

    end = time.time()	# 返回当前时间的时间戳(即从1970年1月1日至今经过的浮点秒数)
    
	# 从用来训练的dataloder里提取数据
    for i, (inputs, _,indexes) in enumerate(trainloader): 
        # measure data loading time
        # 记录数据的加载时间
        data_time.update(time.time() - end)
		
        inputs= inputs.to(device)
        indexes=indexes.to(device)
        optimizer.zero_grad()
        # compute output
        features = model(inputs).float()
        outputs = lemniscate(features, indexes)
        loss = criterion(outputs, indexes)	# 计算对比损失

        train_loss.update(loss.item(), inputs[0].size(0))	# inputs.size(0)表示图片数量
        

       # compute gradient and do SGD step
        # 在 PyTorch 中, 我们需要在开始进行反向传播之前将梯度设置为零,
        # 因为 PyTorch 会在随后的向后传递中累积梯度。
        optimizer.zero_grad()
        # loss.backward(): 为每个具有 require_grad = True 的参数 x 计算 d(loss) / dx。 d(...)/dx是对“...”求导的意思
        # 这些对于每个参数 x 都累积到 x.grad 中。 伪代码: x.grad + = d(loss) / dx
        loss.backward()
        # optimizer.step(): 使用 x.grad 来更新 x 的值。 例如: SGD 优化器执行以下操作: x += -lr * x.grad 
        optimizer.step()

        
        # scaler.scale(loss).backward()
      
        # scaler.step(optimizer) 
        # scaler.update()
        # measure elapsed time
        # 计算训练每一批次花费的时间
        batch_time.update(time.time() - end)
        end = time.time()	# 重置 “批次开始训练的” 时间点
        print('Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
              epoch, i, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))


for epoch in range(start_epoch, start_epoch+200):
    # train(epoch)
    adjust_learning_rate(optimizer, epoch,args)
    train(trainloader, model, criterion, optimizer, epoch, args)
    if epoch%5==0:
        acc = kNN(epoch, model, lemniscate, trainloader, testloader, 200, args.nce_t, 0)
        if acc > best_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'lemniscate': lemniscate,
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            print('==>saving checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc
        print('best accuracy: {:.2f}'.format(best_acc*100))

