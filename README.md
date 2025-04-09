# :page_facing_up: MedShift-SFDA: A Difficulty-graded Benchmark for Source-Free Domain Adaptation in Medical Image Segmentation

### Dependency Preparation
```shell
cd MedShift-SFDA
# Python Preparation
conda create -n MedShift-SFDA python=3.8.5
activate MedShift-SFDA
# (torch 1.7.1+cu110) It is recommended to use the conda installation on the Pytorch website https://pytorch.org/
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```
### Six Methods for Model Training and Inference
- 1. Download the dataset in the paper and modify the relevant path in the configuration file.
- 2. Source Model Train
   -- We use the code provided by [ProSFDA](https://github.com/ShishuaiHu/ProSFDA) to train the source model. If you want to use our trained source model, please contact me.
- 3. Steps to debug six methods: 
```shell
1. DPL Method:
(1) Source code link: [DPL](https://github.com/cchen-cc/SFDA-DPL)
(2) Please click DPL file
(3) Generation phase: generate target domain pseudo-labels
python generate_pseudo.py
(4) Adaptation stage: source model adapts to the target domain
python train_target.py
2. CBMT Method:
(1) Source code link: [CBMT](https://github.com/lloongx/SFDA-CBMT)
(2) Please click CBMT file
(3) Adaptation stage: source model adapts to the target domain
python train_target.py
3. CPR Method:
(1) Source code link: [CPR](https://github.com/xmed-lab/CPR)
(2) Please click CPR file
(3) Generation phase: generate target domain pseudo-labels
python generate_pseudo.py
(4) Adaptation stage: source model adapts to the target domain
Please run them in order:
python sim_learn.py
python pl_refine.py
python train_target.py
4. PCPDL Method:
(1) Source code link: [PCPDL](https://github.com/M4cheal/PCDCL-SFDA)
(2) Please click PCPDL file
(3) Generation phase: generate target domain pseudo-labels
python generate_pseudo.py
(4) Adaptation stage: source model adapts to the target domain
python train_target.py
5. FSM Method:
(1) Source code link: [FSM](https://github.com/CityU-AIM-Group/SFDA-FSM)
(2) Please click on the MedShift-SFDA file and enter the FSM
(3) Generate source-like images
python domain_inversion.py
(4) Adaptation stage: source model adapts to the target domain
python train_adapt.py 
6. ADAMI Method:
(1) Source code link: [ADAMI](https://github.com/mathilde-b/SFDA)
(2) Please click ADAMI file
(3) Adaptation stage: source model adapts to the target domain
python train_target.py
