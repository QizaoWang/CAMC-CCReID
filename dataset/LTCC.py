"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

ACCV 2022 Oral paper: Co-Attention Aligned Mutual Cross-Attention for Cloth-Changing Person Re-Identification
URL: https://openaccess.thecvf.com/content/ACCV2022/html/Wang_Co-Attention_Aligned_Mutual_Cross-Attention_for_Cloth-Changing_Person_Re-Identification_ACCV_2022_paper.html
GitHub: https://github.com/QizaoWang/CAMC-CCReID
"""

import glob
import re

import os.path as osp

from dataset.base_image_dataset import BaseImageDataset


class LTCC(BaseImageDataset):
    """
    LTCC
    Reference:
    Long-Term Cloth-Changing Person Re-identification
    URL: https://naiq.github.io/LTCC_Perosn_ReID.html

    Dataset statistics:
    (1) "train". There are 9,576 images with 77 identities in this folder used for training (46 cloth-change IDs + 31 cloth-consistent IDs).
    (2) "test". There are 7,050 images with 75 identities in this folder used for testing (45 cloth-change IDs + 30 cloth-consistent IDs).
    (3) "query". There are 493 images with 75 identities. We randomly select one query image for each camera and each outfit.
    """

    def __init__(self, dataset_root='data', dataset_filename='LTCC_ReID', verbose=True, **kwargs):
        super(LTCC, self).__init__()
        self.dataset_dir = osp.join(dataset_root, dataset_filename)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')

        self.check_before_run(required_files=[self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir])

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> LTCC loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        pattern = re.compile(r'(\d+)_(\d+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid_container = sorted(list(pid_container))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, _, camid = map(int, pattern.search(img_path).groups())
            assert 0 <= pid <= 151
            assert 1 <= camid <= 12
            camid -= 1  # index starts from 0

            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset