import torch
import utility
import data
import model
import loss
import os
from option import args    # 参数设置
from trainer import Trainer

torch.manual_seed(args.seed)
print('Args object with arguments: {}'.format(args))
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)   # dataset&dataloader
    model = model.Model(args, checkpoint)
    print('Total parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

