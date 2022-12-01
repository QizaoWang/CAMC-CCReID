import time
from tqdm import tqdm
import numpy as np

import torch

from evaluate.metrics import eval_market1501
from utils.util import AverageMeter


def test_CAMC(args, query_loader, gallery_loader, model, use_gpu, ranks=[1, 5, 10], epoch=None):
    batch_time = AverageMeter()
    model.eval()
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(tqdm(query_loader)):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(tqdm(gallery_loader)):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)
        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, args.test_batch))

    # feature normalization
    qf = 1. * qf / (torch.norm(qf, 2, dim=-1, keepdim=True).expand_as(qf) + 1e-12)
    gf = 1. * gf / (torch.norm(gf, 2, dim=-1, keepdim=True).expand_as(gf) + 1e-12)
    # get distmat matrix
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    distmat = distmat.numpy()

    cmc, mAP = eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50)
    if epoch: print("Epoch {}: ".format(epoch), end='')
    print("mAP: {:.4%}  ".format(mAP), end='')
    for r in ranks:
        print("Rank-{:<2}: {:<7.4%}  ".format(r, cmc[r - 1]), end='')
    print()

    return cmc[0], mAP
