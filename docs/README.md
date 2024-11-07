# RBSRNet

## <p align="center">Retinal Bionic Super-Resolution Reconstruction Network: Enhancing Image Details through Hierarchical Feature Fusion</p>

<p align="center">Xiaofen Jia, Junjun Liu, Baiting Zhao, Zhenhuan Liang</p>

## 📖 Introduction:

Drawing inspiration from the retinal information processing mechanism, we propose a novel super-resolution reconstruction method named RBSRNet. This method employs a lateral inhibition module (LIM) to mimic the regulatory mechanism of horizontal cells, enabling local inhibition and enhancement of shallow extracted features, thereby efficiently extracting high-frequency information. Furthermore, by simulating the information integration mechanism of bipolar cells, we design an enhanced hierarchical feature fusion block (EHFFB) and a hierarchical fusion network (HFN). These components facilitate effective fusion and enhancement of features at different levels, improving the model's representational capacity. Experimental results on four datasets demonstrate that RBSRNet significantly outperforms advanced methods, particularly in texture detail recovery. Specifically, RBSRNet achieves an average PSNR that is 0.26dB higher than IRN on the Urban100 dataset, with reconstructed images exhibiting superior visual effects and richer texture details. The source code is available at https://github.com/Ljjsisr/RBSRNet.

<p align="center"><img width="800" alt="image" src="https://github.com/user-attachments/assets/eddf69b4-32a2-4a2e-aa35-97a31f92c791"></p>


## 🔧Installation

This implementation based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Please refer to BasicSR for training and testing. 

```bash
python 3.8.18
pytorch 2.0.0
cuda 11.8
```

```bash
git clone this repo
cd RBSRNet
pip install -r requirements.txt
pip install google-auth-oauthlib
python setup.py develop
```

## 🎈Datasets

Preparing: Please refer to the [Dataset Preparation](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) of BasicSR.

Training: DIV2K or DF2K.

Testing: Set5, Set14, BSD100, Urban100, Manga109 ([Baidu Netdisk](https://pan.baidu.com/s/1NF_McRKPgkRjqFCevjWMiQ?pwd=ci78)).

## ▶️Train and Test

**The complete network code is in (`basicsr/archs`)`**

**The checkpoint file is in the experiments folder**

**The script for running is in options**

- Training with the example option

```bash
python basicsr/train.py -opt options/train/RBSRNet/train_RBSRNet_Lx4.yml 
```

- Testing with the example option

```python basicsr/test.py -opt options/test/RBSRNet/test_RBSRNet_Lx4.yml```

## 📈 Results

![image](https://github.com/user-attachments/assets/3c129bca-16f6-4848-941c-86b3657dc5ac)

![image](https://github.com/user-attachments/assets/f3ae1eba-7f91-4f54-84f1-90ce86bcdcd8)

