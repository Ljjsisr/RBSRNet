# RBSRNet

Drawing inspiration from the retinal information processing mechanism, we propose a novel super-resolution reconstruction method named RBSRNet. This method employs a lateral inhibition module (LIM) to mimic the regulatory mechanism of horizontal cells, enabling local inhibition and enhancement of shallow extracted features, thereby efficiently extracting high-frequency information. Furthermore, by simulating the information integration mechanism of bipolar cells, we design an enhanced hierarchical feature fusion block (EHFFB) and a hierarchical fusion network (HFN). These components facilitate effective fusion and enhancement of features at different levels, improving the model's representational capacity. Experimental results on four datasets demonstrate that RBSRNet significantly outperforms advanced methods, particularly in texture detail recovery. Specifically, RBSRNet achieves an average PSNR that is 0.26dB higher than IRN on the Urban100 dataset, with reconstructed images exhibiting superior visual effects and richer texture details. The source code is available at https://github.com/Ljjsisr/RBSRNet.

This repository is the implementation for the paper "Super-resolution Reconstruction Method of Retinal Bionics for Single Image". 

## 🎈Datasets
Training: DIV2K or DF2K.

Testing: Set5, Set14, BSD100, Urban100, Manga109 ([Baidu Netdisk](https://pan.baidu.com/s/1NF_McRKPgkRjqFCevjWMiQ?pwd=ci78)).

Preparing: Please refer to the [Dataset Preparation](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) of BasicSR.

## ▶️Train and Test
The BasicSR framework is utilized to train our RBSRNet, also testing.

**Training with the example option**

``` python basicsr/train.py -opt options/train/RBSRNet/train_RBSRNet_Lx4.yml ```

**Testing with the example option**

```python basicsr/test.py -opt options/test/RBSRNet/test_RBSRNet_Lx4.yml```

## 💡Results Display

![image](https://github.com/user-attachments/assets/8991ecbb-76de-4488-ad5b-40bbbb2836e3)

