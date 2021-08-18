# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import time
import yaml
import math
import random
import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.distributed as dist

import torchvision.transforms as transforms
import numpy as np
import utils.supernet_utils as nasnet_utils
import matplotlib.pyplot as plt
import matplotlib as mpl

from easydict import EasyDict
from datetime import datetime, timedelta
from tensorboardX import SummaryWriter
from datasets import ImageNetDataset
from utils.distributed_utils import get_rank
from utils.scheduler import CosineLRScheduler
from utils.misc import DistributedSampler, DistributedGivenIterationSampler, AverageMeter, load_state, param_group_no_wd
from utils.distributed_utils import dist_init

plt.rcParams["figure.figsize"] = (13, 10)
plt.style.use('bmh')
mpl.rcParams['figure.dpi'] = 120

parser = argparse.ArgumentParser(description='PyTorch')

parser.add_argument('--config', default='experiments/NSE/config_NSE27.yaml')
parser.add_argument('--balance_s_rate', type=float, default=0.5)
parser.add_argument('--dist_mode', default=False, action='store_true')
parser.add_argument('--load-path', default='', type=str, help='path to checkpoint')


class Env(object):
    def __init__(self):
        self.args = args

        self.start_epoch = 0
        self.iters_per_epoch = 0

        self.batch_size = config.batch_size
        self.workers = config.workers

        self.local_id, self.rank, self.world_size = dist_init()
        assert (self.batch_size % self.world_size == 0)
        assert (self.workers % self.world_size == 0)
        self.batch_size = self.batch_size // self.world_size
        self.workers = self.workers // self.world_size

        assert args.dist_mode  # only distributed trainig is supported

        if self.rank == 0:
            self.tb_logger = SummaryWriter(config.save_path + '/events')

        # >>> seed initialization
        seed = config.get('seed', 233)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        cudnn.benchmark = True

    def get_gpumodel(self, model):
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.local_id],
                                                          output_device=self.local_id, find_unused_parameters=True)
        return model

    def get_model(self):
        raise NotImplementedError(f"not implemented")

    def get_w_optim(self, model):
        if config.optim_W.no_wd:
            params = param_group_no_wd(model)
        else:
            params = model.parameters()

        lr_init = config.optim_W.lr if config.optim_W.warm_epochs == 0 else config.optim_W.lr_warm
        if config.optim_W.optim == 'SGD':
            return torch.optim.SGD(params, lr=lr_init, momentum=config.optim_W.momentum,
                                   weight_decay=config.optim_W.weight_decay)
        elif config.optim_W.optim == 'RMSprop':
            return torch.optim.RMSprop(params, lr=lr_init, alpha=0.9, eps=0.02,
                                       weight_decay=config.optim_W.weight_decay, momentum=config.optim_W.momentum)
        elif config.optim_W.optim == 'SGD_nesterov':
            return torch.optim.SGD(params, lr=lr_init, momentum=config.optim_W.momentum,
                                   weight_decay=config.optim_W.weight_decay, nesterov=True)
        else:
            raise NotImplementedError(f"not supported optimizer: {config.optim_W.optim}")

    def get_w_scheduler(self, optimizer):
        if config.optim_W.decay == 'cos':
            lr_scheduler = CosineLRScheduler(optimizer, self.iters_per_epoch * config.epochs, config.optim_W.lr,
                                             config.optim_W.lr_final, self.iters_per_epoch * config.optim_W.warm_epochs,
                                             config.optim_W.lr_warm,
                                             last_iter=self.iters_per_epoch * self.start_epoch - 1)
            return lr_scheduler
        else:
            raise RuntimeError(f'not implemented lr decay: {config.optim_W.decay}')

    def get_a_optim(self, params):
        if config.optim_A.optim == 'Adam':
            return torch.optim.Adam(params, config.optim_A.lr, betas=(config.optim_A.beta1, config.optim_A.beta2))
        else:
            raise NotImplementedError(f"not implemented optimizer: {config.optim_A.optim}")

    def save_checkpoint(self, state, is_best, round=-1, filename='checkpoint.pth.tar'):
        if self.rank != 0:
            return
        filename = os.path.join(config.save_path, filename)
        torch.save(state, filename)
        if is_best:
            if round >= 0:
                bestname = os.path.join(config.save_path, f'model_final_{round}.pth.tar')
            else:
                bestname = os.path.join(config.save_path, 'model_best.pth.tar')
            shutil.copyfile(filename, bestname)

    def printlogs(self, input):
        if self.rank != 0:
            return
        logname = os.path.join(config.save_path, 'log.txt')
        with open(logname, 'a') as f:
            f.write(input + '\n')

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Env_ImageNet_AHPO(Env):
    def __init__(self):
        super(Env_ImageNet_AHPO, self).__init__()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize({224: 256, 299: 333, 331: 367}[config.input_size]),
            transforms.CenterCrop(config.input_size),
            transforms.ToTensor(),
            normalize,
        ])

        if self.rank == 0:
            print(f"building dataset from {config.train_meta}")
        self.train_dataset = ImageNetDataset(
            config.train_root,
            config.train_meta,
            transform=transform_train,
            read_from='fs')
        self.train_sampler = DistributedSampler(self.train_dataset)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=False, sampler=self.train_sampler, drop_last=True)
        self.iters_per_epoch = len(self.train_loader)

        if self.rank == 0:
            print(f"building dataset from {config.arch_val_meta}")
        self.val_dataset = ImageNetDataset(
            config.train_root,
            config.arch_val_meta,
            transform=transform_test,
            read_from='fs')
        self.val_sampler = DistributedSampler(self.val_dataset, round_up=False)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                                      num_workers=self.workers, pin_memory=False,
                                                      sampler=self.val_sampler, drop_last=False)


    def get_model(self):
        if config.algo == 'NSE':
            model = nasnet_utils.AMBImageNet(scale=config.scale, channel_dist=config.channel_dist,
                                             input_size=config.input_size, alloc_space=config.alloc_space,
                                             cell_plan=config.cell_plan, alloc_plan=config.alloc_plan,
                                             K=config.K)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            raise NotImplementedError("unimplemented algo")
        return model

    @staticmethod
    def get_share_net(alphas, sub_space, alloc_plan, alloc_space=(1, 4, 4, 8, 4)):
        if alloc_plan == 'NR':
            back = ""
            for i in alloc_space:
                back += "N" * i + "R"
            back = back[:-1]
        elif alloc_plan == 'NER':
            back = (lambda x: "N" * x[0] + "R" +
                              "N" * x[1] + "R" +
                              "N" * x[2] + "R" +
                              "N" * x[3] + "E" +
                              "N" * x[4] + "R" +
                              "N" * x[5] + "E")(alloc_space)

        depth = [0] * len(alloc_space)
        valid = [list(alphas[i].argsort()[len(alphas[i]) - sum(alphas[i] > 0):][::-1]) for i in range(len(alphas))]
        depth_index = 0
        for i, branches in enumerate(valid):
            if back[i] == "N":
                if len(branches) > 0:
                    depth[depth_index] += 1
            else:
                depth_index += 1

        final = [[] for _ in range(len(valid))]
        for i, branches in enumerate(valid):
            for branch in branches:
                final[i].append(sub_space[i].tolist()[branch])

        final = [i for i in final if len(i) > 0]

        return final, depth

    def is_pareto_efficient_simple(self, costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient

    def pretty_print_front(self, code, alloc_plan, alloc_space=(1, 4, 4, 8, 4)):  # retrieve architecture code in list
        if alloc_plan == 'NR':
            back = ""
            for i in alloc_space:
                back += "N" * i + "R"
            back = back[:-1]
        elif alloc_plan == 'NER':
            back = (lambda x: "N" * x[0] + "R" +
                              "N" * x[1] + "R" +
                              "N" * x[2] + "R" +
                              "N" * x[3] + "E" +
                              "N" * x[4] + "R" +
                              "N" * x[5] + "E")(alloc_space)

        depth = [0] * len(alloc_space) if not 'I' in alloc_plan else [0] * len(alloc_space[0])
        if 'E' in alloc_plan:
            depth = depth + [0] * 2

        depth_index = 0
        for i, branches in enumerate(code):
            if back[i] in ["N", "D", "I"]:
                if len(branches) > 0:
                    depth[depth_index] += 1
            else:
                depth_index += 1
        code = [i for i in code if len(i) > 0]

        return code, depth

    def train_model(self):
        if not os.path.exists(config.save_path) and self.rank == 0:
            print("env make dir: " + config.save_path)
            try:
                os.makedirs(config.save_path)
            except Exception as e:
                print(e)
                pass
        self.printlogs(str(config))
        self.printlogs(str(args))
        model = self.get_model()
        netpara = model.netpara
        self.printlogs('  Total params: %.2fM' % (netpara))
        model = self.get_gpumodel(model)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = self.get_w_optim(model)
        arch_optimizer = self.get_a_optim(model.module.arch_parameters)

        start_round = 0
        if args.load_path:
            states = load_state(args.load_path, model, optimizer, a_optim=self.get_a_optim)
            if states is not None:
                self.start_epoch, arch_optimizer, start_round = states
            args.load_path = None

        if config.get("contind", False):  # second space init | final optimized search space states which delivers NSENet-27

            model.module.prev_best = [[0], [9, 6], [0, 3], [0], [], [16, 13], [3], [9, 12], [9], [17, 10, 11, 13],
                                      [9, 2], [9, 15], [12], [1, 10, 6], [9, 12], [15], [0, 9], [17, 13, 14], [10, 21],
                                      [3, 19], [13], [3, 19, 14, 7]]
            model.module.prev_bests = [
                [[], [6], [3], [0], [], [16], [3], [], [9], [11, 13, 17], [9], [15], [], [6], [9, 12], [], [9],
                 [17, 14], [21], [19], [13], [19, 3, 14]],
                [[], [6], [3], [0], [], [16, 13], [3], [], [9], [11, 17], [9, 2], [15], [], [6], [9, 12], [15], [9],
                 [17, 14], [21], [19], [], [19, 3, 14]],
                [[], [6, 9], [3], [0], [], [13], [], [9], [9], [11, 17], [9], [15], [], [1, 6], [12], [], [9], [17, 14],
                 [21], [19], [13], [19, 3, 14]],
                [[0], [6], [3], [0], [], [16], [3], [12], [9], [10, 11, 17], [9, 2], [], [], [1, 6], [12], [15], [9],
                 [17, 13, 14], [21], [19], [13], [19, 3, 14]],
                [[], [6, 9], [3, 0], [0], [], [16], [3], [], [9], [11, 17], [9, 2], [15], [], [10, 6], [9, 12], [15],
                 [], [17, 14], [21], [3, 19], [13], [19, 3, 14]],
                [[0], [6, 9], [3], [0], [], [16], [3], [], [9], [11, 17], [9, 2], [15], [], [6], [9, 12], [], [0, 9],
                 [17, 14], [21], [19], [13], [19, 3, 14]],
                [[], [6], [3], [0], [], [16, 13], [], [9], [9], [11, 17], [9], [15], [12], [1, 6], [12], [], [9],
                 [17, 14], [21], [19], [13], [19, 3, 14]],
                [[0], [6, 9], [3, 0], [], [], [16], [3], [], [9], [11, 17], [9, 2], [15], [], [1, 6], [12], [], [0, 9],
                 [17, 13, 14], [21], [19], [13], [19, 3, 14]],
                [[0], [6, 9], [3, 0], [], [], [13], [3], [9], [9], [11, 17], [9], [15], [], [1, 6], [9, 12], [], [9],
                 [17, 13, 14], [10, 21], [19], [13], [19, 3, 14]],
                [[], [6, 9], [3], [0], [], [13], [3], [9], [9], [11, 17], [9, 2], [15], [], [1, 6], [9, 12], [], [9],
                 [17, 14], [21], [19], [13], [19, 3, 7, 14]],
                [[0], [6], [3], [], [], [16], [3], [9], [9], [11, 17], [9], [9, 15], [], [10, 6], [12], [], [9],
                 [17, 14], [21], [19], [], [19, 3, 14]],
                [[0], [6, 9], [3], [0], [], [13], [3], [], [9], [11, 17], [9, 2], [15], [], [6], [9, 12], [], [9],
                 [17, 14], [10, 21], [19], [13], [19, 7, 14]],
                [[0], [6], [3], [0], [], [13], [3], [9], [9], [11, 17], [9, 2], [15], [], [1, 6], [9, 12], [], [],
                 [17, 14], [21], [], [13], [19, 3, 14]],
                [[0], [6], [3, 0], [0], [], [16], [3], [], [9], [11, 17], [9, 2], [15], [12], [1, 6], [12], [], [],
                 [17, 14], [21], [19], [], [19, 3, 14]],
                [[], [6], [3, 0], [], [], [16], [3], [9], [9], [11, 17], [9, 2], [15], [], [6], [12], [], [9], [17, 14],
                 [10, 21], [19], [], [19, 3, 14]],
                [[0], [6], [3], [], [], [13], [3], [9], [9], [11, 17], [9, 2], [], [], [6], [9, 12], [15], [], [17, 14],
                 [21], [19], [13], [19, 3, 14]],
                [[0], [6], [3], [0], [], [16], [3], [], [9], [11, 17], [9, 2], [], [12], [1, 6], [9, 12], [], [9],
                 [17, 14], [21], [3], [13], [19, 3, 14]]]  # <--- NSENET-27
            model.module.pending_space = []
            for _, _ in enumerate(model.module.cells):
                curr_order = torch.tensor(np.random.permutation(45) + 27).cuda()
                dist.broadcast(curr_order, src=0)
                curr_order = curr_order.tolist()
                model.module.pending_space.append(curr_order)
            model.module.update_space()


        for round in range(model.module.total_rounds):
            if round < start_round:
                continue
            if round == config.get('max_rounds', None):
                self.printlogs('search finished\n')
                return

            if config.get('balance', False):
                model.module.clean_space(config.drop_threshold, preserve=config.get('preserve_prev', True))
            self.printlogs(
                f"Search Round [{round + 1}/{model.module.total_rounds}], Sub search space: {list(map(lambda x: x.tolist(), model.module.sub_space))}")

            lr_scheduler = self.get_w_scheduler(optimizer)

            for epoch in range(0, config.epochs):
                if epoch < self.start_epoch:
                    continue
                # train for one epoch
                self.train(model, criterion, optimizer, arch_optimizer, lr_scheduler, epoch, round)

                #  save
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'arch_params': model.module.arch_parameters,
                    'optimizer': optimizer.state_dict(),
                    'arch_optimizer': arch_optimizer.state_dict(),
                    'round': round,
                    'pending_space': model.module.pending_space,
                    'sub_space': model.module.sub_space,
                    'prev_space': model.module.prev_best,
                    'prev_archs': model.module.prev_bests
                }, epoch + 1 == config.epochs, round=round)

            front_stat, cost_stat = [], []
            sample_num = config.get('sample_num', 500)
            extra_sample_num = config.get('extra_sample_num', 100)
            extra_sample_drift = config.get('extra_sample_drift', 10)
            if config.latency_alpha != 0:
                model.module.batch_sampler(sample_num, 72, 6, extra_sample_num, 2, latency=True,
                                           non_alpha=config.get("uniform_sample", False),
                                           balance_s_rate=args.balance_s_rate)
            else:
                model.module.batch_sampler(sample_num, config.mac_target + config.get("mac_target_calibration", 0),
                                           config.mac_drift, extra_sample_num,
                                           extra_sample_drift, non_alpha=config.get("uniform_sample", False),
                                           balance_s_rate=args.balance_s_rate)

            top1_before, top1_after, top1_diff = [], [], []
            top1_before_stat, top1_after_stat = [], []
            for index in range(sample_num + extra_sample_num):
                this_code = model.module.get_next_pin_paths()
                curr_top1, curr_mac, curr_lat = self.validate(model, index, bulk=True)
                top1_before.append(curr_top1)

                front_stat.append([curr_mac if config.latency_alpha == 0 else curr_lat.item(), curr_top1])
                cost_stat.append([curr_mac if config.latency_alpha == 0 else curr_lat.item(), -curr_top1])
                self.printlogs(f"Corresponding model code: {this_code}")
                self.printlogs(f"Corresponding raw model code: {model.module.paths_pin}")

            if round > 0:
                for index in range(len(model.module.prev_bests)):
                    this_code = model.module.get_next_prev_pin_paths()
                    prev_top1, prev_mac, prev_lat = self.validate(model, index, prev=True)
                    top1_before.append(prev_top1), top1_before_stat.append(prev_top1)

                    front_stat.append([prev_mac if config.latency_alpha == 0 else prev_lat.item(), prev_top1 + 1e-10])
                    cost_stat.append([prev_mac if config.latency_alpha == 0 else prev_lat.item(), -prev_top1 + 1e-10])
                    self.printlogs(f"Corresponding prev_best model code: {this_code}")
                    self.printlogs(f"Corresponding raw prev_best model code: {model.module.prev_paths_pin}")

            front_stat, cost_stat = np.array(front_stat), np.array(cost_stat)
            pareto_front = self.is_pareto_efficient_simple(cost_stat)

            ######### plot pareto front
            if self.rank == 0:
                fig, _ = plt.subplots(1, 1)
                plt.scatter(*list(zip(*(front_stat[pareto_front]))))
                plt.scatter(*list(zip(*(front_stat[~pareto_front]))))
                self.tb_logger.add_figure(f'Pareto Front: top-1_mac/lat', fig, round)
            ######### print pareto front architectures
            pareto_front[sample_num:sample_num + extra_sample_num] = False
            assert sum(pareto_front[sample_num:]) == sum(pareto_front[sample_num + extra_sample_num:])
            model.module.set_front(pareto_front)
            self.printlogs(
                f"Old Pareto Points: {sum(pareto_front[sample_num:])}, New Pareto Points: {sum(pareto_front[:sample_num])}")
            for i in range(sum(pareto_front[:sample_num])):
                self.printlogs(
                    f"New Pareto Front Point: {front_stat[pareto_front][i]} \ncode: {self.pretty_print_front(model.module.prev_bests[i], config.alloc_plan, config.alloc_space)}")
            for i in range(sum(pareto_front[sample_num:])):
                self.printlogs(
                    f"Old Pareto Front Point: {front_stat[pareto_front][i + sum(pareto_front[:sample_num])]} \ncode: {self.pretty_print_front(model.module.prev_bests[i + sum(pareto_front[:sample_num])], config.alloc_plan, config.alloc_space)}")
            ###########################

            model.module.update_space()
            if round + 1 == model.module.total_rounds or round + 1 == config.get('max_rounds', None):
                self.printlogs(f"Final Result: {list(map(lambda x: x.tolist(), model.module.sub_space))}")

            model.module.re_init()

            optimizer = self.get_w_optim(model)

            arch_optimizer = self.get_a_optim(model.module.arch_parameters)

            self.start_epoch = 0

            self.save_checkpoint({
                'epoch': 0,
                'state_dict': model.state_dict(),
                'arch_params': model.module.arch_parameters,
                'optimizer': optimizer.state_dict(),
                'arch_optimizer': arch_optimizer.state_dict(),
                'round': round + 1,
                'pending_space': model.module.pending_space,
                'sub_space': model.module.sub_space,
                'prev_space': model.module.prev_best,
                'prev_archs': model.module.prev_bests
            }, is_best=False, filename=f'model_final_sampled_{round}.pth.tar')

        self.printlogs('search finished\n')
        return

    def train(self, model, criterion, optimizer, arch_optimizer, lr_scheduler, epoch, rounds):
        self.train_sampler.set_epoch(epoch)
        val_sampler = DistributedGivenIterationSampler(self.val_dataset,
                                                       math.ceil(len(self.train_dataset) * 1.0 / self.batch_size),
                                                       self.batch_size, seed=epoch)
        local_val_loader = iter(torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                                            num_workers=self.workers, pin_memory=False,
                                                            sampler=val_sampler))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        val_batch_time = AverageMeter()
        val_data_time = AverageMeter()
        val_losses = AverageMeter()
        val_latency_losses = AverageMeter()
        val_param_losses = AverageMeter()
        val_mac_losses = AverageMeter()
        val_top1 = AverageMeter()
        val_top5 = AverageMeter()

        # switch to train mode
        model.train()
        model.module.sp_val_flag = False

        end = time.time()
        for i, (input, target) in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            curr_iter = epoch * self.iters_per_epoch + i + rounds * config.epochs * self.iters_per_epoch
            real_curr_iter = epoch * self.iters_per_epoch + i
            lr_scheduler.step(real_curr_iter)
            current_lr = lr_scheduler.get_lr()[0]

            if config.get('balance', False):
                model.module.clean_space(config.drop_threshold, preserve=config.get('preserve_prev', True))

            if not config.pure_arch_opt and (not config.get('merged_arch_update', False) or i % config.Aiter != 0):
                target = target.squeeze().cuda().long()
                input = input.cuda()

                # weight update

                output, latency, param, mac = model(input, balance=config.get('balance', False),
                                                    balance_s_rate=args.balance_s_rate)
                loss = criterion(output, target)

                # measure accuracy and record loss
                prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))

                loss /= self.world_size

                reduced_loss = loss.data.clone()
                reduced_prec1 = prec1.clone() / self.world_size
                reduced_prec5 = prec5.clone() / self.world_size

                dist.all_reduce(reduced_loss)
                dist.all_reduce(reduced_prec1)
                dist.all_reduce(reduced_prec5)

                losses.update(reduced_loss.item(), input.size(0))
                top1.update(reduced_prec1.item(), input.size(0))
                top5.update(reduced_prec5.item(), input.size(0))

                if self.rank == 0:
                    print(
                        f'sample forward latency: {(latency + model.module.get_back_time()).detach().cpu().item()}\tparams: {param + model.module.get_back_param()}M\tmacs: {mac + model.module.get_back_mac()}M')
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                dist.barrier()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                remain_secs = (self.iters_per_epoch * config.epochs - real_curr_iter) * batch_time.avg
                remain_time = timedelta(seconds=round(remain_secs))
                finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
                if i % config.print_freq == 0 or config.get('merged_arch_update',
                                                            False) and i % config.print_freq == config.print_freq - 1:
                    if self.rank == 0:
                        self.tb_logger.add_scalar('Train Loss', losses.val, curr_iter)
                        self.tb_logger.add_scalar('Train Top1', top1.val, curr_iter)
                        self.tb_logger.add_scalar('Train Top5', top5.val, curr_iter)
                        self.tb_logger.add_scalar('LR', current_lr, curr_iter)
                    self.printlogs(f'Epoch: [{epoch}][{i}/{self.iters_per_epoch}]\t'
                                   f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                   f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                   f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                                   f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                   f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                                   f'Learning rate: {current_lr:.4f}'
                                   f'Supernet Remaining Time {remain_time} ({finish_time})')

                end = time.time()

            if i % config.Aiter != 0:
                continue
            #################################################################################

            # importance indicator update

            val_input, val_target = (input, target) if config.get('merged_arch_update', False) else next(local_val_loader)
            val_data_time.update(time.time() - end)

            val_target = val_target.squeeze().cuda().long()
            val_input = val_input.cuda()

            output, latency, param, mac = model(val_input, arch_update=True, round=rounds,
                                                strict_prev=config.get('strict_prev', False),
                                                arch_sample_balance=config.get('arch_sample_balance', False))
            loss = criterion(output, val_target)
            prec1, prec5 = self.accuracy(output.data, val_target, topk=(1, 5))

            full_lat = latency + model.module.get_back_time()
            latency_losses = ((full_lat - config.latency_target) ** config.latency_beta) * config.latency_alpha
            full_para = param + model.module.get_back_param()
            param_losses = ((full_para - config.param_target) ** config.param_beta) * config.param_alpha
            full_mac = mac + model.module.get_back_mac()
            mac_losses = ((full_mac - config.mac_target) ** config.mac_beta) * config.mac_alpha

            loss /= self.world_size
            reduced_loss = loss.data.clone()
            reduced_prec1 = prec1.clone() / self.world_size
            reduced_prec5 = prec5.clone() / self.world_size

            dist.all_reduce(reduced_loss)
            dist.all_reduce(reduced_prec1)
            dist.all_reduce(reduced_prec5)

            val_losses.update(reduced_loss.item(), input.size(0))
            val_latency_losses.update(latency_losses.item(), input.size(0))
            val_param_losses.update(param_losses.item(), input.size(0))
            val_mac_losses.update(mac_losses.item(), input.size(0))
            val_top1.update(reduced_prec1.item(), input.size(0))
            val_top5.update(reduced_prec5.item(), input.size(0))

            final_loss = loss + latency_losses / self.world_size
            final_loss += param_losses / self.world_size
            final_loss += mac_losses / self.world_size

            # compute gradient and do SGD step
            if self.rank == 0:
                print(
                    f'val expected latency: {full_lat.detach().cpu().item()}\tparams: {full_para.detach().cpu().item()}M\tmacs: {full_mac.detach().cpu().item()}M')

            arch_optimizer.zero_grad()

            final_loss.backward()

            for params in model.module.arch_parameters:
                if type(params) == list:
                    for param in params:
                        dist.all_reduce(param.grad.data, async_op=True)
                else:
                    dist.all_reduce(params.grad.data, async_op=True)

            dist.barrier()

            arch_optimizer.step()

            # measure elapsed time
            val_batch_time.update(time.time() - end)

            remain_secs = (self.iters_per_epoch * config.epochs - real_curr_iter) * \
                          (val_batch_time.avg / config.Aiter + batch_time.avg)
            remain_time = timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))

            if i % config.print_freq == 0:
                if self.rank == 0:
                    self.tb_logger.add_scalar('Val Loss', val_losses.val, curr_iter)
                    self.tb_logger.add_scalar('Val Latency Loss', val_latency_losses.val, curr_iter)
                    self.tb_logger.add_scalar('Val Param Loss', val_param_losses.val, curr_iter)
                    self.tb_logger.add_scalar('Val Mac Loss', val_mac_losses.val, curr_iter)
                    self.tb_logger.add_scalar('Val Latency ', full_lat.detach().cpu().item(), curr_iter)
                    self.tb_logger.add_scalar('Val Param ', full_para.detach().cpu().item(), curr_iter)
                    self.tb_logger.add_scalar('Val Mac ', full_mac.detach().cpu().item(), curr_iter)
                    self.tb_logger.add_scalar('Val Top1', val_top1.val, curr_iter)
                    self.tb_logger.add_scalar('Val Top5', val_top5.val, curr_iter)
                self.printlogs(f'Epoch: [{epoch}][{i}/{self.iters_per_epoch}]\t'
                               'Arch update\t'
                               f'Time {val_batch_time.val:.3f} ({val_batch_time.avg:.3f})\t'
                               f'Data {val_data_time.val:.3f} ({val_data_time.avg:.3f})\t'
                               f'Loss {val_losses.val:.4f} ({val_losses.avg:.4f})\t'
                               f'Latency Loss {val_latency_losses.val:.4f} ({val_latency_losses.avg:.4f})\t'
                               f'Param Loss {val_param_losses.val:.4f} ({val_param_losses.avg:.4f})\t'
                               f'Mac Loss {val_mac_losses.val:.4f} ({val_mac_losses.avg:.4f})\t'
                               f'Prec@1 {val_top1.val:.3f} ({val_top1.avg:.3f})\t'
                               f'Prec@5 {val_top5.val:.3f} ({val_top5.avg:.3f})'
                               f'Indicator Remaining Time {remain_time} ({finish_time})')
                if config.algo in ["NSE"]:
                    alphas = model.module.arch_parameters[0].detach()
                    sub_alphas = []
                    if self.rank == 0:
                        for ith, alpha in enumerate(alphas):
                            if len(model.module.sub_space[ith]) > 0:
                                sub_alpha = alpha.index_select(0, model.module.sub_space[ith]).cpu().numpy()
                            else:
                                sub_alpha = np.array([-2])
                            self.tb_logger.add_histogram(f'alpha_{ith}', sub_alpha, curr_iter)
                            sub_alphas.append(sub_alpha)
                    self.printlogs(f'alphas: {alphas}')

                    self.printlogs(f'sub_alphas:')
                    for sa in sub_alphas:
                        self.printlogs(f'{list(sa)}')
                    self.printlogs(
                        f'OPs with indicator > 0: {self.get_share_net(sub_alphas, model.module.sub_space, config.alloc_plan, config.alloc_space)}')
                else:
                    raise NotImplementedError("unimplemented algo")

            end = time.time()

    def validate(self, model, index, prev=False, bulk=False, skip_cali=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.module.sp_val_flag = True
        # Do not switch to evaluate mode
        # model.eval()   We shall not use the eval mode as the running mean is volatile for supernet

        criterion = nn.CrossEntropyLoss().cuda()

        end = time.time()
        with torch.no_grad():
            if bulk:
                model.module.pin_paths()
            elif prev:
                model.module.pin_prev_paths()

            if not skip_cali:
                model.train()
                for i in range(config.get('val_fiter', 50)):
                    input, target = next(self.train_loader)
                    input = input.cuda()
                    output, latency, param, mac = model(input, balance=True, prev_val=prev, bulk_val=bulk)
                    if i % 40 == 0:
                        self.printlogs(f'PreTest: [{i}/{config.get("val_fiter", 50)}]\t')

            model.eval()
            for i, (input, target) in enumerate(self.val_loader):
                input = input.cuda()
                target = target.squeeze().view(-1).cuda().long()
                # compute output
                output, latency, param, mac = model(input, balance=True, prev_val=prev, bulk_val=bulk)

                # measure accuracy and record loss
                prec1, prec5 = self.accuracy(output.data, target, topk=(1, 5))
                loss = criterion(output, target)

                num = input.size(0)
                losses.update(loss.item(), num)
                top1.update(prec1.item(), num)
                top5.update(prec5.item(), num)

                # measure elapsed time
                batch_time.update(time.time() - end)

                if i % 20 == 0:
                    self.printlogs(f'Test: [{i}/{len(self.val_loader)}]\t'
                                   f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                   f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                                   f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                   f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')
                end = time.time()

        total_num = torch.Tensor([losses.count])
        top1_sum = torch.Tensor([top1.avg * top1.count])
        top5_sum = torch.Tensor([top5.avg * top5.count])

        dist.all_reduce(total_num)
        dist.all_reduce(top1_sum)
        dist.all_reduce(top5_sum)

        final_top1 = top1_sum.item() / total_num.item()
        final_top5 = top5_sum.item() / total_num.item()

        model.module.sp_val_flag = False

        self.printlogs(
            f'*{"prev" if prev else "sample"} index: {index} val forward latency: {(latency + model.module.get_back_time()).detach().cpu().item()}\tparams: {param + model.module.get_back_param()}M\tmacs: {mac + model.module.get_back_mac()}M')
        self.printlogs(
            f' *{"prev" if prev else "sample"} index: {index} Prec@1 {final_top1:.3f} Prec@5 {final_top5:.3f}')

        return final_top1, mac + model.module.get_back_mac(), latency + model.module.get_back_time()


if __name__ == '__main__':
    global args, config

    args = parser.parse_args()
    print(args)

    with open(args.config) as f:
        config = EasyDict(yaml.load(f))

    config.save_path = os.path.join(os.path.dirname(args.config),
                                    f'search_{config.exp_name}_' + datetime.now().strftime("%F-%T"))
    env = Env_ImageNet_AHPO()
    env.train_model()


def print_debug(*args, **kw):
    return
    if get_rank() == 0:
        print(*args, **kw)
