from __future__ import print_function, absolute_import

from dataset.Market1501 import Market1501
from dataset.CelebreID import CelebreID
from dataset.LTCC import LTCC


__img_factory = {
    'market1501': Market1501,
    'celeb': CelebreID,
    'ltcc': LTCC,
}


def get_dataset(args):
    name = args.dataset
    if name not in __img_factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __img_factory.keys()))

    dataset = None
    if name == 'market1501':
        dataset = __img_factory[name](dataset_root=args.dataset_root, dataset_filename=args.dataset_filename,
                                      split_id=args.split_id, cuhk03_labeled=args.cuhk03_labeled, cuhk03_classic_split=args.cuhk03_classic_split,)
    elif name in ['celeb', 'ltcc']:
        dataset = __img_factory[name](dataset_root=args.dataset_root, dataset_filename=args.dataset_filename,)
    return dataset