"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

ACCV 2022 Oral paper: Co-Attention Aligned Mutual Cross-Attention for Cloth-Changing Person Re-Identification
URL: https://openaccess.thecvf.com/content/ACCV2022/html/Wang_Co-Attention_Aligned_Mutual_Cross-Attention_for_Cloth-Changing_Person_Re-Identification_ACCV_2022_paper.html
GitHub: https://github.com/QizaoWang/CAMC-CCReID
"""

from __future__ import absolute_import
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Data
    # dataset
    parser.add_argument('-d', '--dataset', type=str, default='celeb')
    parser.add_argument('--dataset_root', type=str, default='data', help="root path to data directory")
    parser.add_argument('--dataset_filename', type=str, default='Celeb-reID')
    # data augmentation
    parser.add_argument('--height', type=int, default=256,
                        help="height of an image (default: 256)")
    parser.add_argument('--width', type=int, default=128,
                        help="width of an image (default: 128)")
    parser.add_argument('--horizontal_flip_pro', type=float, default=0.5,
                        help="Random probability for image horizontal flip")
    parser.add_argument('--pad_size', type=int, default=10,
                        help="Value of padding size")
    parser.add_argument('--random_erasing_pro', type=float, default=0.5,
                        help="Random probability for image random erasing")
    # data manager and loader
    parser.add_argument('--split_id', type=int, default=0, help="split index")
    parser.add_argument('--train_batch', default=64, type=int,
                        help="train batch size")
    parser.add_argument('--num_instances', type=int, default=4,
                        help="number of instances per identity")
    parser.add_argument('-j', '--num_workers', default=8, type=int,
                        help="number of data loading workers")
    parser.add_argument('--test_batch', default=128, type=int,
                        help="test batch size")
    # CUHK03-specific setting
    parser.add_argument('--cuhk03_labeled', action='store_true',
                        help="whether to use labeled images, if false, detected images are used (default: False)")
    parser.add_argument('--cuhk03_classic_split', action='store_true',
                        help="whether to use classic split by Li et al. CVPR'14 (default: False)")
    parser.add_argument('--use_metric_cuhk03', action='store_true',
                        help="whether to use cuhk03-metric (default: False)")

    # Optimization options
    # epoch
    parser.add_argument('--start_epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--max_epoch', default=150, type=int,
                        help="maximum epochs to run")

    # lr
    parser.add_argument('--lr', default=0.0003, type=float,
                        help="initial learning rate")
    parser.add_argument('--warm_up_factor', default=0.1, type=float,
                        help="warm up factor")
    parser.add_argument('--warm_up_method', default="linear", type=str,
                        choices=['constant', 'linear'],
                        help="warm up factor")
    parser.add_argument('--warm_up_epochs', default=10, type=int,
                        help="take how many epochs to warm up")

    parser.add_argument('--step_size', default=40, type=int,
                        help="step size to decay learning rate (>0 means this is enabled)")
    parser.add_argument('--step_milestones', default=[40, 80], nargs='*', type=int,
                        help="epoch milestones to decay learning rate, multi steps")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")
    parser.add_argument('--weight_decay', default=5e-04, type=float,
                        help="lr weight decay")
    parser.add_argument('--weight_decay_bias', default=0.0005, type=float,
                        help="lr weight decay for layers with bias")

    # Architecture/ Model
    parser.add_argument('--backbone_pretrain', type=bool, default=True,
                        help='whether use pretrained backbone (default True)')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='pre-trained model path')
    parser.add_argument('--pose_net_path', type=str, default='', metavar='PATH',
                        help='pre-trained pose model path')

    # Miscs/ Others
    # cpu/gpu and seed
    parser.add_argument('--use_cpu', action='store_true', help="use cpu")
    parser.add_argument('--gpu_devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=666, help="manual seed")

    parser.add_argument('--evaluate', action='store_true', help="evaluation only")
    # for training
    parser.add_argument('--start_eval_epoch', type=int, default=0,
                        help="start to evaluate after training a specific epoch")
    parser.add_argument('--eval_epoch', type=int, default=10,
                        help="run evaluation for every N epochs (set to -1 to test after training)")

    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--save_checkpoint', action='store_true', help='save model checkpoint')

    args = parser.parse_args()
    return args


def print_args(args):
    print('------------------------ Args -------------------------')
    for k, v in vars(args).items():
        print('%s: %s' % (k, v))
    print('--------------------- Args End ------------------------')
    return
