import os
import math
from decimal import Decimal
import time

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)    #  self.scheduler.last_epoch 0
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()                                                   #  after   --->    self.scheduler.last_epoch 1
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1                                   # epoch = 2
        #lr = self.scheduler.get_last_lr()[0]
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        tqdm_train = tqdm(enumerate(self.loader_train))
        for batch, (lr, hr, _) in tqdm_train:  # lr, hr, index
            # lr, hr = self.prepare([lr, hr])      # to tensor
            lr=lr.to('cuda',non_blocking=True)
            hr=hr.to('cuda',non_blocking=True)

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, self.scale)
            loss = self.loss(sr, hr)
            if loss.item() < self.args.skip_threshold * self.error_last:   # 1e6*1e8
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t timer_model: {:.2f} + timer_data: {:.2f}s'.format(    # model and data usage time seconds
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        # epoch = self.scheduler.last_epoch + 1  # ????????????1????????????2
        epoch = self.scheduler.last_epoch   # ????????????1????????????2
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test): 
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        # lr, hr = self.prepare([lr, hr])
                        lr=lr.to('cuda',non_blocking=True)
                        hr=hr.to('cuda',non_blocking=True)
                    else:
                        # lr = self.prepare([lr])[0]
                        lr=lr.to('cuda',non_blocking=True)

                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)   # ????????????????????????

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)   #psnr
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(   # like:  [Set5 x4]       PSNR: 29.322 (Best: 29.322 @epoch 1)
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n Now time: {}\n'.format(timer_test.toc(), time.asctime( time.localtime(time.time()) ) ), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))   # -->ckp --> checkpoint = utility.checkpoint(args)

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device,non_blocking=True)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

