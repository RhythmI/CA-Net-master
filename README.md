# CA-Net
Code for paper: Context-aware Network for Semi-supervised Segmentation of 3D Left Atrium

Our code is origin from [UA-MT](https://github.com/yulequan/UA-MT) and [SASS-Net](https://github.com/kleinzcy/SASSnet)

You can find paper in [UA-MT](https://arxiv.org/abs/2007.10732), [SASS-Net](https://ojs.aaai.org/index.php/AAAI/article/view/17066)

## Requirements
Some important required packages include:
* [Pytorch][torch_link] version >=0.4.1.
* TensorBoardX
* Python >= 3.6
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......


[torch_link]:https://pytorch.org/

# Usage

1. Clone the repo:
```
git clone https://github.com/RhythmI/CA-Net.git
cd CA-Net-master
```
2. Put the data in `data/2018LA_Seg_Training Set`.

3. Train the model
```
cd code
python train_CANet.py
```

4. Test the model
```
python test_LA.py
```
Our best model are saved in model dir.

# Citation

If you find our work is useful for you, please cite us.
