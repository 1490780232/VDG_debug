from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class SpCLTrainer_UDA(object):
    def __init__(self, encoder, memory, source_classes):
        super(SpCLTrainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes

    def train(self, epoch, data_loader_source, data_loader_target,
                    optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, _, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()
            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)
            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)
            # forward
            f_out = self._forward(inputs)

            # de-arrange batch
            f_out = f_out.view(device_num, -1, f_out.size(-1))
            f_out_s, f_out_t = f_out.split(f_out.size(1)//2, dim=1)
            f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)), f_out_t.contiguous().view(-1, f_out.size(-1))

            # compute loss with the hybrid memory
            loss_s = self.memory(f_out_s, s_targets)
            loss_t = self.memory(f_out_t, t_indexes+self.source_classes)

            loss = loss_s+loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_s {:.3f} ({:.3f})\t'
                      'Loss_t {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class SpCLTrainer_USL(object):
    def __init__(self, encoder, memory):
        super(SpCLTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.ce = nn.CrossEntropyLoss()
    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, pids, indexes = self._parse_data(inputs)
            # forward
            f_out= self._forward(inputs)
            # loss_ce = self.ce(s_out, pids)
            # compute loss with the hybrid memory
            loss_memory = self.memory(f_out, pids) #indexes
            loss = loss_memory  #loss_ce+
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.memory.updata_features(f_out.detach(), pids)
            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)



class VDGTrainer_USL(object):
    def __init__(self, encoder, memory):
        super(VDGTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.ce = nn.CrossEntropyLoss()
    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            # process inputs
            inputs, pids, indexes = self._parse_data(inputs)
            bs = len(pids)
            inputs = torch.cat(inputs, dim=0)
            pids = torch.cat([pids,pids], dim=0)
            # forward
            f_out= self._forward(inputs)
            loss_memory = self.memory(f_out, pids)
            loss = loss_memory
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.memory.updata_features(f_out.detach()[:bs] , pids[:bs]) #
            self.memory.updata_features(f_out.detach()[:bs] , pids[:bs]) #
            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return [imgs[0].cuda(), imgs[1].cuda()], pids.cuda(), indexes.cuda()
        # return imgs[0].cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)



class VDGTrainer_USL_view(object):
    def __init__(self, encoder, memory):
        super(VDGTrainer_USL, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.ce = nn.CrossEntropyLoss()
    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, percam_tempV = [],  concate_intra_class = []):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            # process inputs
            inputs, pids,views, indexes = self._parse_data(inputs)
            bs = len(pids)
            inputs = torch.cat(inputs, dim=0)
            pids = torch.cat([pids,pids], dim=0)
            views = torch.cat(views, dim=0)
            # forward
            f_out= self._forward(inputs)
            loss_memory = self.memory(f_out, pids)
            loss_view=0
            if epoch>=30:
                for cc in torch.unique(views):
                # print(cc)
                    inds = torch.nonzero(views == cc).squeeze(-1)
                    percam_targets = pids[inds]
                    # print(percam_targets)
                    percam_feat = f_out[inds]
                    associate_loss = 0
                    target_inputs = torch.matmul(F.normalize(percam_feat), F.normalize(percam_tempV.t().clone()))
                    temp_sims = target_inputs.detach().clone()
                    target_inputs /= 0.07 
                    for k in range(len(percam_feat)):
                        ori_asso_ind = torch.nonzero(concate_intra_class == percam_targets[k]).squeeze(-1)
                        temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                        sel_ind = torch.sort(temp_sims[k])[1][-50:]
                        concated_input = torch.cat((target_inputs[k, ori_asso_ind], target_inputs[k, sel_ind]), dim=0)
                        concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                            torch.device('cuda'))
                        concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                        associate_loss += -1 * (
                                F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                            0)).sum()
                    loss_view += 0.1 * associate_loss / len(percam_feat)
            loss = loss_memory + loss_view
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.memory.updata_features(f_out.detach()[:bs] , pids[:bs]) #
            self.memory.updata_features(f_out.detach()[:bs] , pids[:bs]) #
            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, views, indexes = inputs
        return [imgs[0].cuda(), imgs[1].cuda()], pids.cuda(), [views[0].cuda(),views[1].cuda()],indexes.cuda()
        # return imgs[0].cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
