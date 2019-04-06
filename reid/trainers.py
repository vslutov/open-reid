from __future__ import print_function, absolute_import
import time
import os.path as osp
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
from tqdm.autonotebook import tqdm

from .evaluation_metrics import accuracy
from .loss import OIMLoss, TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, name):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.name = name

    def train(self, epoch, data_loader, optimizer, print_freq=10):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        runs_dir = osp.join('runs', self.name + '_' + current_time)
        writer = SummaryWriter(log_dir=runs_dir)

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.data.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                writer.add_scalars(
                  'loss',
                  {'train_id_loss': losses.avg
                  },
                  epoch)

                writer.add_scalars(
                  'id_accuracy',
                  {'train_id_accuracy': precisions.avg
                  },
                  epoch)

                losses.reset()
                precisions.reset()

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = [Variable(imgs)]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(*inputs)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            loss = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec.item()
        elif isinstance(self.criterion, OIMLoss):
            loss, outputs = self.criterion(outputs, targets)
            prec, = accuracy(outputs.data, targets.data)
            prec = prec.item()
        elif isinstance(self.criterion, TripletLoss):
            loss, prec = self.criterion(outputs, targets)
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return loss, prec
