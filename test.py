import torch
from option import args    # 参数设置3
from PIL import Image
from torchvision import transforms
from model.gcsr import GCSR
import model

torch.manual_seed(args.seed)
print('Args object with arguments: {}'.format(args))



net = GCSR(args)
net = net.cuda()
x = torch.randn(1,3,20,20)
x = x.cuda()

res = net(x)
print(res.shape)




