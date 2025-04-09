# :page_facing_up: MedShift-SFDA: A Difficulty-graded Benchmark for Source-Free Domain Adaptation in Medical Image Segmentation

### Dependency Preparation
```shell
cd MedShift-SFDA
# Python Preparation
conda create -n ESFDA_SFDA python=3.8.5
activate MedShift-SFDA
# (torch 1.7.1+cu110) It is recommended to use the conda installation on the Pytorch website https://pytorch.org/
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```
### Six Methods for Model Training and Inference
- 1. Download the dataset in the paper and modify the relevant path in the configuration file.
- 2. Source Model Train
   -- We use the code provided by [ProSFDA](https://github.com/ShishuaiHu/ProSFDA) to train the source model. If you want to use our trained source model, please contact me.
