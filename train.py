"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

ACCV 2022 Oral paper: Co-Attention Aligned Mutual Cross-Attention for Cloth-Changing Person Re-Identification
URL: https://openaccess.thecvf.com/content/ACCV2022/html/Wang_Co-Attention_Aligned_Mutual_Cross-Attention_for_Cloth-Changing_Person_Re-Identification_ACCV_2022_paper.html
GitHub: https://github.com/QizaoWang/CAMC-CCReID
"""

import time
from tqdm import tqdm

from utils.util import AverageMeter


def train_CAMC(train_loader, model, class_criterion, optimizer, scheduler, use_gpu):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_load_time = AverageMeter()
    end = time.time()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        imgs, pids, _ = data
        data_load_time.update(time.time() - end)  # measure data loading time

        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()

        model.train()
        ys = model(imgs)

        loss = class_criterion(ys, pids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # track time and loss
        batch_time.update(time.time() - end)
        end = time.time()
        losses.update(loss.item(), pids.size(0))

    scheduler.step()
