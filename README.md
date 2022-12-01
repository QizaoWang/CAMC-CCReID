# Co-Attention Aligned Mutual Cross-Attention for Cloth-Changing Person Re-Identification

> Official PyTorch implementation of ["Co-Attention Aligned Mutual Cross-Attention for Cloth-Changing Person Re-Identification"](https://openaccess.thecvf.com/content/ACCV2022/papers/Wang_Co-Attention_Aligned_Mutual_Cross-Attention_for_Cloth-Changing_Person_Re-Identification_ACCV_2022_paper.pdf). (ACCV 2022 Oral)
>
> Qizao Wang, Xuelin Qian, Yanwei Fu, Xiangyang Xue
>
> Fudan University



## Getting Started

### Environment

- Python == 3.8
- PyTorch == 1.12.1

### Prepare Data

Please download cloth-changing person Re-ID datasets and place them in any path `DATASET_ROOT`. Take [Celeb-reID](https://github.com/Huang-3/Celeb-reID) as an example:

    DATASET_ROOT
    	└─ Celeb-reID
    		├── train
    		├── query
    		├── gallery

### Training

```sh
python main.py --gpu_devices 0 --pose_net_path POSE_NET_PATH --dataset celeb --dataset_root DATASET_ROOT --dataset_filename Celeb-reID --save_dir SAVE_DIR --save_checkpoint
```

`--pose_net_path ` : replace `POSE_NET_PATH` with the path of pretrained [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) weights (download [here](https://drive.google.com/file/d/10ZfIsFgReAGdDwOZSoc4URPM7dt4EEXz/view?usp=sharing))

`--dataset_root ` : replace `DATASET_ROOT` with your dataset root path

`--save_dir`: replace `SAVE_DIR ` with the path to save log files and checkpoints

### Evaluation

```sh
python main.py --gpu_devices 0 --pose_net_path POSE_NET_PATH --dataset celeb --dataset_root DATASET_ROOT --dataset_filename Celeb-reID --resume RESUME_PATH --save_dir SAVE_DIR --evaluate
```

`--resume`: replace `RESUME_PATH ` with the path of the saved checkpoint

### Results

- **Celeb-reID**

| Backbone  | Rank-1 | Rank-5 | mAP  |
| :-------: | :----: | :----: | :--: |
| ResNet-50 |  57.5  |  71.5  | 12.3 |

- **LTCC**

| Backbone  |    Setting     | Rank-1 | mAP  |
| :-------: | :------------: | :----: | :--: |
| ResNet-50 | Cloth-Changing |  36.0  | 15.4 |
| ResNet-50 |    Standard    |  73.2  | 35.3 |

You can achieve similar results with released codes.

## Citation

Please cite the following paper in your publications if it helps your research:

```
@InProceedings{Wang_2022_ACCV,
    author    = {Wang, Qizao and Qian, Xuelin and Fu, Yanwei and Xue, Xiangyang},
    title     = {Co-Attention Aligned Mutual Cross-Attention for Cloth-Changing Person Re-Identification},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {December},
    year      = {2022},
    pages     = {2270-2288}
}
```



## Contact

Any questions or discussions are welcome!

Qizao Wang (<qzwang22@m.fudan.edu.cn>)