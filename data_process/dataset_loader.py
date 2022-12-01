from __future__ import print_function, absolute_import

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.util import read_image
from data_process import transforms as T
from data_process import samplers


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid


def get_dataset_loader(dataset, args, use_gpu):
    # data augmentation
    transform_train = T.Compose([
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        T.Resize((args.height, args.width)),
        T.RandomHorizontalFlip(p=args.horizontal_flip_pro),
        T.Pad(padding=args.pad_size),
        T.RandomCrop((args.height, args.width)),

        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=args.random_erasing_pro, mean=[0.0, 0.0, 0.0]),
    ])

    transform_test = T.Compose([
        T.Resize((args.height, args.width)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # sampler
    sampler = samplers.RandomIdentitySampler(dataset.train, batch_size=args.train_batch,
                                             num_instances=args.num_instances)

    # dataloader
    pin_memory = use_gpu
    train_loader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=sampler,
        batch_size=args.train_batch, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=True,
    )

    query_loader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    gallery_loader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.num_workers,
        pin_memory=pin_memory, drop_last=False,
    )

    return train_loader, query_loader, gallery_loader
