"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

ACCV 2022 Oral paper: Co-Attention Aligned Mutual Cross-Attention for Cloth-Changing Person Re-Identification
URL: https://openaccess.thecvf.com/content/ACCV2022/html/Wang_Co-Attention_Aligned_Mutual_Cross-Attention_for_Cloth-Changing_Person_Re-Identification_ACCV_2022_paper.html
GitHub: https://github.com/QizaoWang/CAMC-CCReID
"""

from __future__ import absolute_import
import sys
import time
import datetime
import os
import os.path as osp
import numpy as np

import torch
from torch import nn

from utils.arguments import get_args, print_args
from utils.util import set_random_seed, Logger, load_checkpoint, save_checkpoint
from data_process import data_manager, dataset_loader
from scheduler.warm_up_multi_step_lr import WarmupMultiStepLR
from loss.cross_entropy_loss import CrossEntropyLabelSmooth
from models.CAMC import CAMC
from train import train_CAMC
from test import test_CAMC


def main():
    args = get_args()
    # log
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print_args(args)

    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
    else:
        print("Currently using CPU (GPU is highly recommended)")

    set_random_seed(args.seed, use_gpu)

    # dataset
    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.get_dataset(args)

    # dataset loader
    train_loader, query_loader, gallery_loader = \
        dataset_loader.get_dataset_loader(dataset, args=args, use_gpu=use_gpu)

    num_classes = dataset.num_train_pids
    model = CAMC(args, num_classes=num_classes, pretrain=args.backbone_pretrain)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    if use_gpu: model = nn.DataParallel(model).cuda()

    # loss
    class_criterion = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=use_gpu)

    # optimizer and scheduler
    params = [{'params': model.module.parameters() if use_gpu else model.parameters(), 'lr': args.lr}]
    optimizer = torch.optim.Adam(params, weight_decay=args.weight_decay)
    scheduler = WarmupMultiStepLR(optimizer, milestones=args.step_milestones, gamma=args.gamma,
                                  warmup_factor=args.warm_up_factor, warmup_iters=args.warm_up_epochs,
                                  warmup_method=args.warm_up_method, last_epoch=args.start_epoch - 1)

    # only test
    if args.evaluate:
        print("Evaluate only")
        test_CAMC(args, query_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None)
        return 0

    # train
    print("==> Start training")
    start_time = time.time()
    train_time = 0
    best_mAP, best_rank1 = -np.inf, -np.inf
    best_epoch_mAP, best_epoch_rank1 = 0, 0
    for epoch in range(args.start_epoch, args.max_epoch):
        start_train_time = time.time()
        train_CAMC(train_loader, model, class_criterion, optimizer, scheduler, use_gpu)
        train_time += round(time.time() - start_train_time)

        # evaluate
        if (epoch + 1) > args.start_eval_epoch and args.eval_epoch > 0 and (epoch + 1) % args.eval_epoch == 0 \
                or (epoch + 1) == args.max_epoch:
            print("==> Test")
            rank1, mAP = test_CAMC(args, query_loader, gallery_loader,
                                   model, use_gpu, ranks=[1, 5, 10], epoch=epoch + 1)

            is_best_mAP = mAP > best_mAP
            is_best_rank1 = rank1 > best_rank1
            if is_best_mAP:
                best_mAP = mAP
                best_epoch_mAP = epoch + 1
            if is_best_rank1:
                best_rank1 = rank1
                best_epoch_rank1 = epoch + 1

            # save
            if args.save_checkpoint:
                state_dict = model.module.state_dict() if use_gpu else model.state_dict()
                save_checkpoint({
                    'state_dict': state_dict,
                    'rank1': rank1,
                    'mAP': mAP,
                    'epoch': epoch,
                }, is_best_mAP, is_best_rank1, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) +
                                                        '_mAP_' + str(round(mAP * 100, 2)) + '_rank1_' + str(
                    round(rank1 * 100, 2)) + '.pth'))

    print("==> Best mAP {:.4%}, achieved at epoch {}".format(best_mAP, best_epoch_mAP))
    print("==> Best Rank-1 {:.4%}, achieved at epoch {}".format(best_rank1, best_epoch_rank1))

    # time using info
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))


if __name__ == '__main__':
    args = get_args()
    main()
