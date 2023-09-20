# Visual Realism Assessment for Face-swap Videos
This is the code of [this](https://arxiv.org/abs/2302.00918) ICIG2023 paper.



## Usage

### Get data
The DFGC2022 dataset can be download from [GoogleDrive](https://drive.google.com/file/d/1FYeEGHvShszO97RnjXtCV-lK3q1qeV51/view?usp=share_link) or [BaiduCloud](https://pan.baidu.com/s/1dUmHl3iGjVxDu47fH9JwrA?pwd=wuhp). Note that the train-test seperation in the above data (*'train'*, *'test1'*, *'test2'*, and *'test3'*) is the one applied to the DFGC-VRA competetion (IJCB2023), which is different from the seperation applied in this paper (*'C1'*, *'C2'*, *'C3'*). The labels used in this paper can be found in the `metadata` folder. The downloaded dataset is assumed to be saved in the `data` folder.

For video preprocessing, we crop the videos and keep only the facial region. Run `video_crop.py` to preform cropping. Note that the `cv2.CascadeClassifier` method is used, so you might need to download the corresponding setting file.

### Extra features
We borrowed models from related tasks as our feature extractors. You may find these models according to the table below.

For the traditional-feature-based models, you can get the features with modifying a few paths in their original code. For the deep models, we provide the feature extraction code in the `extra_feature` folder. You need to download the models first and put it in the folder with the corrseponding name.

All extracted features are assumed to be solved in the `feature` folder.

|method|link|
|----|----|
|BRISQUE|[matlab](https://github.com/vztu/VIDEVAL/tree/master/features/initial_feature_set)|
|GM-LOG|[matlab](https://github.com/vztu/VIDEVAL/tree/master/features/initial_feature_set)|
|FRIQUEE|[matlab](https://github.com/vztu/VIDEVAL/tree/master/features/initial_feature_set)|
|TLVQM|[matlab](https://github.com/vztu/VIDEVAL/tree/master/features/initial_feature_set)|
|V-BLIINDS|[pytorch](https://github.com/pavancm/vbliinds)|
|VIDEVAL|[matlab](https://github.com/vztu/VIDEVAL/tree/master)|
|RAPIQUE|[matlab](https://github.com/vztu/RAPIQUE)|
|ResNet|Provided by torchvision|
|VGGFace|[pytorch](https://github.com/prlz77/vgg-face.pytorch)|
|DFDC-ispl|[pytorch](https://github.com/polimi-ispl/icpr2020dfdc)|
|DFGC-1st|[pytorch](https://github.com/chenhanch/DFGC-2022-1st-place)|

### Feature selection
We first decide the dimension of the selected features by running `feats_num_select.py`, then perform the selection by running `feats_select.py`.

### Train and test
We train a SVR on the selected features and test its performance. Run `predict.py` to get a prediction, run `eval.py` for evaluation. Please refer to the code for more details. 

This scrip is mostly borrowed from [here](https://github.com/vztu/BVQA_Benchmark.git)[<sup>1</sup>](#refer-anchor-1).


## Reference

<div id="refer-anchor-1"></div>

- [1] Tu, Zhengzhong, et al. "UGC-VQA: Benchmarking blind video quality assessment for user generated content." IEEE Transactions on Image Processing 30 (2021): 4449-4464.

