# RBSRNet

## <p align="center">Retinal Bionic Super-Resolution Reconstruction Network: Enhancing Image Details through Hierarchical Feature Fusion</p>

<p align="center">Xiaofen Jia, Junjun Liu, Baiting Zhao, Zhenhuan Liang</p>

## 📖 Introduction:

Drawing inspiration from the retinal information processing mechanism, we propose a novel super-resolution reconstruction method named RBSRNet. This method employs a lateral inhibition module (LIM) to mimic the regulatory mechanism of horizontal cells, enabling local inhibition and enhancement of shallow extracted features, thereby efficiently extracting high-frequency information. Furthermore, by simulating the information integration mechanism of bipolar cells, we design an enhanced hierarchical feature fusion block (EHFFB) and a hierarchical fusion network (HFN). These components facilitate effective fusion and enhancement of features at different levels, improving the model's representational capacity. Experimental results on four datasets demonstrate that RBSRNet significantly outperforms advanced methods, particularly in texture detail recovery. Specifically, RBSRNet achieves an average PSNR that is 0.26dB higher than IRN on the Urban100 dataset, with reconstructed images exhibiting superior visual effects and richer texture details. The source code is available at https://github.com/Ljjsisr/RBSRNet.

<img width="465" alt="image" src="https://github.com/user-attachments/assets/b7293251-6d1d-451c-a0c6-d878987e51f6">

## 🔧Installation

The BasicSR framework is utilized to train our RBSRNet, also testing.

- 完整的 BasicSR 中文解读文档 PDF，你所需要的内容可以在相应的章节中找到。

- 文档的最新版可以从 [BasicSR-docs/releases](https://github.com/XPixelGroup/BasicSR-docs/releases) 下载。

## 🎈Datasets
Training: DIV2K or DF2K.

Testing: Set5, Set14, BSD100, Urban100, Manga109 ([Baidu Netdisk](https://pan.baidu.com/s/1NF_McRKPgkRjqFCevjWMiQ?pwd=ci78)).

Preparing: Please refer to the [Dataset Preparation](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) of BasicSR.

## ▶️Train and Test

**Training with the example option**

```bash
python basicsr/train.py -opt options/train/RBSRNet/train_RBSRNet_Lx4.yml 
```

**Testing with the example option**

```python basicsr/test.py -opt options/test/RBSRNet/test_RBSRNet_Lx4.yml```

## 💡Results Display

![image](https://github.com/user-attachments/assets/8991ecbb-76de-4488-ad5b-40bbbb2836e3)

