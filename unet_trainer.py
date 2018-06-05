import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
from unet_models import unet11
from unet_models import Loss
import numpy as np
import pytz
import scipy.misc
from torch.autograd import Variable
import tqdm
import fcn
import utils


def cyclic_lr(epoch, init_lr=1e-4, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))
    return lr

class Trainer(object):

    def __init__(self, cuda, model, optimizer,use_resnet,
                 train_loader, val_loader, out, max_iter,deep_sup_factor,
                 sz_average=False, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.opt = optimizer
        self.use_resnet=use_resnet
        self.deep_sup_factor=deep_sup_factor
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.sz_average = sz_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        #self.optim = self.opt(cyclic_lr(self.epoch))
        self.optim=self.opt
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)


            score,deep_score= self.model(data)
            score=torch.squeeze(score,1)

            creterion=Loss()
            loss = creterion(score, target)

            #if np.isnan(float(loss.data[0])):
            #    raise ValueError('loss is nan while validating')
            #val_loss += float(loss.data[0]) / len(data)
            val_loss+=float(loss.data[0])
            imgs = data.data.cpu()
            mask1=(F.sigmoid(score.data))>0.5
            mask2=(F.sigmoid(score.data))<=0.5
            score.data[mask1]=1
            score.data[mask2]=0
            lbl_pred = score.data.cpu().numpy()[:, :, :]
            lbl_pred=lbl_pred.astype(int)
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                '''
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
                '''
        metrics = utils.label_accuracy_score(
            label_trues, label_preds, n_class)
        '''
        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))
        '''
        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        if self.use_resnet:
        	torch.save({
            	'epoch': self.epoch,
            	'iteration': self.iteration,
            	'arch': self.model.__class__.__name__,
            	#'optim_state_dict': self.optim.state_dict(),
            	'encoder_model_state_dict': self.model.encoder.state_dict(),
            	'decoder_model_state_dict': self.model.decoder.state_dict(),
            	'best_mean_iu': self.best_mean_iu,
        	}, osp.join(self.out, 'checkpoint.pth.tar'))
        else:
        	torch.save({
            	'epoch': self.epoch,
            	'iteration': self.iteration,
            	'arch': self.model.__class__.__name__,
            	#'optim_state_dict': self.optim.state_dict(),
            	'model_state_dict':self.model.state_dict(),
            	'best_mean_iu': self.best_mean_iu,
        	}, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))
            print('best_mean_iu={}'.format(self.best_mean_iu))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            #lr=cyclic_lr(self.epoch)
            #self.optim=self.opt(lr)
            if self.use_resnet:
                for optimizer in self.optim:
                    optimizer.zero_grad()
            else:
                self.optim.zero_grad()

            if self.deep_sup_factor>0:
                score,deep_score=self.model(data)
                score=torch.squeeze(score,1)
                deep_score=torch.squeeze(deep_score,1)
            else:
                score = self.model(data)
                score=torch.squeeze(score,1)

            creterion=Loss()
            loss=creterion(score,target)
            if self.deep_sup_factor>0:
                loss=loss+self.deep_sup_factor*creterion(deep_score,target)
            #loss /= len(data)
            #if np.isnan(float(loss.data[0])):
            #    raise ValueError('loss is nan while training')
            loss.backward()
            if self.use_resnet:
                for optimizer in self.optim:
                    optimizer.step()
            else:
                self.optim.step()
            #self.optim.step()


            metrics = []
            mask1=(F.sigmoid(score.data))>0.5
            mask2=(F.sigmoid(score.data))<=0.5
            score.data[mask1]=1
            score.data[mask2]=0
            lbl_pred = score.data.cpu().numpy()[:, :, :]
            lbl_pred=lbl_pred.astype(int)
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                utils.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss.data[0]] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
