<h1 align="center">
    <b>NA-ExU-Net</b>
</h1>

<h2 align="center">
    Noise-aware Extended U-Net with Split Encoder and Feature Refinement Module for Robust Speaker Verification in Noisy Environments
</h2>

<h3 align="left">
	<p>
	<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
	<a href="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-21-04.html#rel-21-04"><img src="https://img.shields.io/badge/21.04-2496ED?style=for-the-badge&logo=Docker&logoColor=white"></a>
	<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white">
	</p>
</h3>

This repository offers source code for the following paper:

* **Title** : Noise-aware Extended U-Net with Split Encoder and Feature Refinement Module for Robust Speaker Verification in Noisy Environments (Accepted for publication in IEEE Access)
* **Authors** : Chan-yeong Lim, Jungwoo Heo, Ju-ho Kim, Hyun-seo Shin, and Ha-Jin Yu

### Paper abstract
<img src="https://github.com/chan-yeong0519/NA-ExU-Net/blob/main/NA-ExU-Net_framework.PNG" width="1000" height="350">
Speech data gathered from real-world environments typically contain noise, a significant element that undermines the performance of deep neural network-based speaker verification (SV) systems. To mitigate performance degradation due to noise and develop noise-robust SV systems, several researchers have integrated speech enhancement (SE) and SV systems. We previously proposed the extended U-Net (ExU-Net), which achieved state-of-the-art performance in SV in noisy environments by jointly training SE and SV systems. In the SE field, some studies have shown that recognizing noise components within speech can improve the system's performance. Inspired by these approaches, we propose a noise-aware ExU-Net (NA-ExU-Net) that acknowledges noise information in the SE process based on the ExU-Net architecture. The proposed system comprises a Split Encoder and a feature refinement module (FRM). The Split Encoder handles the speech and noise separately by dividing the encoder blocks, whereas FRM is designed to inhibit the propagation of irrelevant data via skip connections. To validate the effectiveness of our proposed framework in noisy conditions, we evaluated the models on the VoxCeleb1 test set with added noise from the MUSAN corpus. The experimental results demonstrate that NA-ExU-Net outperforms the ExU-Net and other baseline systems under all evaluation conditions. Furthermore, evaluations in out-of-domain noise environments indicate that NA-ExU-Net significantly surpasses existing frameworks, highlighting its robustness and generalization capabilities. 

# 1. Prerequisites
## 1.1. Environment Setting

* We used 'nvcr.io/nvidia/pytorch:21.04-py3' image of Nvidia GPU Cloud for conducting our experiments
* The details of the environment settings can be found in 'Dockerfile' file.
* Run 'build.sh' file to make docker image
```
./scripts/build.sh
```
(We conducted experiment using 4 NVIDIA RTX 3090 GPUs)

## 1.2. Datasets
* We used VoxCeleb1 dataset for training and test
* For evaluating the model in noisy conditions, we utilized MUSAN and Nonspeech100 dataset.

## 2. Run experiment
Set experimental arguments in `arguments.py` file. Here is list of system arguments to set.

```python
1. 'usable_gpu': '' # ex) '0,1,2,3'
  'usable_gpu' is the GPU which is used in the experiment.

2. 'path_...': ''
  'path_...' is the path where ... dataset is stored.
  'path_logging' is the path of saving experiments.
  The Input type is str
```

Then, just run main.py in NAExUnet_code!
```python
python main.py
```

# Citation
Please cite this paper if you make use of the code. 
```
@article{lim2024noise,
  title={Noise-aware Extended U-Net with Split Encoder and Feature Refinement Module for Robust Speaker Verification in Noisy Environments},
  author={Lim, Chan-Yeong and Heo, Jungwoo and Kim, Ju-Ho and Shin, Hyun-Seo and Yu, Ha-Jin},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```
