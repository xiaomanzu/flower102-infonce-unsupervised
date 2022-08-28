# flower102-infonce-unsupervised
## 数据集的下载与处理
这里采用的是flower102数据集，在102flowers里有相关flower图片,同时还有`imagelabels.mat`和`setid.mat`文件，分类的代码在`data_prepare.py`中，可根据`文件存放界面.jpg`在编译器里建好文件夹，图片处理好后可自动分类为train、test、valid三类
## 预模型训练
该模型借用了instdisc模型，采用infonce loss作为目标函数进行计算，其中正样本来自图片本身，负样本是除本身之外的所有图片，采用resnet50模型进行训练，对train数据集进行无标签学习
`pre-train.py`
## 微调 
将预训练后的权重保存下来，冻结其他的参数只对最后一层全连接层fc进行更新，将valid和test带入训练，均可得到80%的准确率
`main.py`
