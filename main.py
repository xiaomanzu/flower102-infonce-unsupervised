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
import shutil

from NCEAverage import NCEAverage
from LinearAverage import LinearAverage
from NCECriterion import NCECriterion
from utils import AverageMeter
from test import NN, kNN, Test
model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch flower102 Training')
parser.add_argument('--lr', default=0.06, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
# parser.add_argument('--r', dest='resume',
#                       help='resume checkpoint or not',
#                       default=True, type=bool)
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--nce-k', default=-1, type=int,
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
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')

args = parser.parse_args()
scaler = torch.cuda.amp.GradScaler()
global best_acc1

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
# Data loading code
# 把训练集和测试集放进来
data_dir = '/workspace/flower102/prepare_pic'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

transform_train = transforms.Compose([transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.CenterCrop(224),#从中心开始裁剪，留下224*224的。（随机裁剪得到的数据更多）
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率去翻转，0.5就是50%翻转，50%不翻转
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),#参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
        transforms.RandomGrayscale(p=0.025),#概率转换成灰度率，3通道就是R=G=B
        transforms.ToTensor(),  #转成tensor的格式
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])#均值，标准差（拿人家算好的）
    ])
   
transform_valid=transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #要和训练集保持一致的标准化操作
    ])
    
batch_size=256
best_prec1=0


trainset = Flower102Instance(os.path.join(train_dir),transform=transform_train) 
trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=2)
validset =Flower102Instance(os.path.join(valid_dir),transform=transform_valid) 
validloader = DataLoader(validset, shuffle=False, batch_size=batch_size, num_workers=2)
ndata = trainset.__len__()  


if args.gpu is not None:
    print("Use GPU: {} for training".format(args.gpu))

print('==> Building model..')
#使用resnet152的网络结构args.arch
model = torchvision.models.__dict__['resnet50'](128)
# model = models.resnet50(pretrained=True)
fc_inputs = model.fc.in_features
model.fc=nn.Linear(fc_inputs, 102)
# model.fc = nn.Sequential(
#     nn.Linear(fc_inputs, 256),
#     # nn.ReLU(),
#     # nn.Dropout(0.4),
#     # nn.Linear(256, 102)
# )
# model.fc.weight = nn.Parameter()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model=model.cuda()
    cudnn.benchmark = True
model.to(device)
# if isinstance(model,torch.nn.DataParallel):
# 		model = model.module

# define leminiscate
if args.nce_k > 0:
    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
else:
    lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)
# define loss function
if hasattr(lemniscate, 'K'):
    criterion = NCECriterion(ndata)
else:
    criterion = nn.CrossEntropyLoss()

lemniscate.to(device)
criterion.to(device)
# freeze all layers but the last fc
# 冻结 Encoder 的参数不更新，而只更新最后分类器的参数
    
# optimize only the linear classifier
# 只优化线性分类器# 优化器 optimizer 作用的参数为 parameters, 它只包含分类器 fc 的 weight 和 bias 这两部分
# for layer in model.fc.modules():
#     if isinstance(layer,nn.Linear):
print(model.fc.parameters)
optimizer = torch.optim.SGD(model.fc.parameters(), args.lr,momentum=0.9,weight_decay=5e-4)
# parameters = list(filter(lambda p: p.requires_grad, layer.parameters()))
# assert len(parameters) == 2  # fc.weight, fc.bias
# optionally resume from a checkpoint
  
# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/'+'ckpt.t7')
model.load_state_dict(checkpoint['model'],strict=False)
lemniscate = checkpoint['lemniscate']
best_acc = checkpoint['acc']
start_epoch = 0
print("=> loaded checkpoint '{}' (epoch {})"
    .format(args.resume, checkpoint['epoch']))


for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:	
        param.requires_grad = False
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()
cudnn.benchmark = True


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
def train(trainloader,model, criterion, optimizer, epoch, args):
    
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    # 切换到模型的训练模式
    model.train()

    end = time.time()	# 返回当前时间的时间戳(即从1970年1月1日至今经过的浮点秒数)
    
	# 从用来训练的dataloder里提取数据
    for i, (inputs, targets,indexes) in enumerate(trainloader): 
        # measure data loading time
        # 记录数据的加载时间
        data_time.update(time.time() - end)
        inputs= inputs.to(device)
        indexes=indexes.to(device)
        optimizer.zero_grad()
        # compute output
        # 计算 q 与 k 的输出结果
        # output, target = model(im_q=images[0], im_k=images[1])
        # loss = criterion(output, target)	# 计算对比损失
        features = model(inputs).float()
        features=features.to(device)
        loss = criterion(features, targets.cuda())

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # Top-1/Top-5是针对于 K 种分类的准确度, 并会记录“对比损失”
        train_loss.update(loss.item(), inputs[0].size(0))	# inputs.size(0)表示图片数量
        

       # compute gradient and do SGD step
        # 在 PyTorch 中, 我们需要在开始进行反向传播之前将梯度设置为零,
        # 因为 PyTorch 会在随后的向后传递中累积梯度。
        optimizer.zero_grad()
        # loss.backward(): 为每个具有 require_grad = True 的参数 x 计算 d(loss) / dx。 d(...)/dx是对“...”求导的意思
        # 这些对于每个参数 x 都累积到 x.grad 中。 伪代码: x.grad + = d(loss) / dx
        # loss.backward()
        # optimizer.step(): 使用 x.grad 来更新 x 的值。 例如: SGD 优化器执行以下操作: x += -lr * x.grad 
        # optimizer.step()

        scaler.scale(loss).backward()
      
        scaler.step(optimizer) 
        scaler.update()
        # measure elapsed time
        # 计算训练每一批次花费的时间
        batch_time.update(time.time() - end)
        end = time.time()	# 重置 “批次开始训练的” 时间点
        print('Epoch: [{}][{}/{}]'
              'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
              epoch, i, len(trainloader), batch_time=batch_time, data_time=data_time, train_loss=train_loss))

if args.evaluate:
    validate(validloader, model, criterion, args)
# return
for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch,args)

        # train for one epoch
        train(trainloader, model, criterion, optimizer, epoch,args)

        # evaluate on validation set
        prec1 = Test(epoch, model, validloader)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)



