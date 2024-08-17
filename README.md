# Visual Realism Assessment for Face-swap Videos
This is the code of [this](https://arxiv.org/abs/2302.00918) ICIG2023 paper.



## Usage

### Get data
To download the dataset of this work, please refer to the [DFGC-2022 project](https://github.com/NiCE-X/DFGC-2022) and apply with an application form.The downloaded dataset is assumed to be saved in the `data` folder.

The labels used in this paper can be found in the `metadata` folder. The `C*-mos` files provide the ground truth realism score of each video, while the `C*-mos-subid` files provide a method level realism score by averaging the scores of all videos with the same submission id, as each submission id repersents a unique face-swap method.

For video preprocessing, we crop the videos and keep only the facial region. Relative coordinates of the bounding box for each video is provided in the `crop` folder, with the top left corner set as the origin. `x` and `y` denote the location of the center point of the bounding box. Note that the width `w` and height `h` of the boxes in the given files are the original face detection results given by `cv2.CascadeClassifier`, so you will need to crop the image using a 1.3 times larger box (`w*1.3 × h*1.3`) to reproduce the experiment in the paper.

### Extra features
We borrowed models from related tasks as our feature extractors. You may find these models according to the table below.

For the traditional-feature-based models, you can get the features with modifying a few paths in their original code. For the deep models, we provide the feature extraction code in the `extra_feature` folder. You need to download the models first and put it in the folder with the corrseponding name.

All extracted features are assumed to be saved in the `feature` folder.

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
We first decide the dimensionality of the selected features by running `feats_num_select.py`, then perform the selection by running `feats_select.py`.

### Train and test
We train a SVR on the selected features and test its performance. Run `predict.py` to get a prediction, run `eval.py` for evaluation. Please refer to the code for more details. 

This scrip is mostly borrowed from [here](https://github.com/vztu/BVQA_Benchmark.git).


## Cite
If you find our work useful, please cite it as:
```
@article{sun2023visual,
  title={Visual Realism Assessment for Face-swap Videos},
  author={Sun, Xianyun and Dong, Beibei and Wang, Caiyong and Peng, Bo and Dong, Jing},
  journal={arXiv preprint arXiv:2302.00918},
  year={2023}
}
```

