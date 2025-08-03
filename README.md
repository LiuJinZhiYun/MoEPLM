# PSMOE: A computational method for predicting the solubility of protein and the effect of protein mutation on the solubility of protein 

## Complete Directory Structure  
├── identification/  # predicting protein solubility
│   ├── data/
│   │   ├── train.fasta
│   │   ├── val.fasta
│   │   ├── esmc600m_all_data.emb #run Sol_ESMC.py to get it.
│   │   └── test.fasta
│   ├── output/      # save trained model.pth
│   ├── model.py              
│   ├── dataset.py   
│   ├── metrics.py   
│   ├── Sol_ESMC.py   
│   ├── train.py   
│   ├── test.py   
│   └── utils.py            
│
├── mutation/  # predicting protein mutation effects on solubility
│   ├── data/
│   │   ├── train.xlsx
│   │   ├── train.xlsx
│   │   ├── wild_type_60.emb #run Mut_ESMC.py to get it.
│   │   ├── mutation_type_60.emb #run Mut_ESMC.py to get it.
│   │   └── test.xlsx
│   ├── output/      # save trained model.pth
│   ├── model.py              
│   ├── dataset.py   
│   ├── metrics.py   
│   ├── Mut_ESMC.py   
│   ├── train.py   
│   ├── test.py   
│   └── utils.py   

## Installation Guide
```bash
git clone https://github.com/LiuJinZhiYun/PSMOE.git
cd PSMOE
```
## Dependencies  
This package is tested with Python 3.10 + PyTorch 2.1.2+cu118 + CUDA 11.8  Run the following to create a conda environment and install the required Python packages (modify `pytorch-cuda=11.8` according to your CUDA version).
```bash
conda create -n PSMOE python=3.10
conda activate PSMOE
```
```bash
pip install numpy==1.26.4 pandas==2.2.3 matplotlib==3.10.1 scikit-learn==1.6.1
```

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```
  
```bash
pip install biopython==1.85 esm==3.1.1
```

```bash
pip install termcolor==2.1.0 tqdm==4.67.1 filelock==3.13.1 
```
 
```bash
pip install openpyxl==3.1.5 tokenizers==0.13.3 huggingface_hub==0.30.4 
```
 
```bash
pip install --root-user-action=ignore \
  /root/autodl-tmp/causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  /root/autodl-tmp/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
causal_conv1d-1.4.0...and mamba_ssm-2.2.2...are downloaded from [https://github.com/Dao-AILab/causal-conv1d/releases] and [https://github.com/Dao-AILab/causal-conv1d/issues/9]
![0f606210bfd871cb397601572f35c20d](https://github.com/user-attachments/assets/dcb4c073-255f-4564-ad4a-9da19738f619)


The example provides versions of  mamba_ssm compatible with PyTorch 2.1.2 (CUDA 11.8) and Python 3.10.
 For other versions, 
download causal_conv1d and mamba_ssm from their respective GitHub releases:
- mamba_ssm: [https://github.com/state-spaces/mamba/releases](https://github.com/state-spaces/mamba/releases)
- causal_conv1d: [https://github.com/Dao-AILab/causal-conv1d/releases](https://github.com/Dao-AILab/causal-conv1d/releases)
## Illustration of config in model.py
```bash
class MoEConfig:
    def __init__(self):
        self.num_experts_per_tok = 2 # For each sample, the model will choose the topk expert
        self.n_routed_experts = 4 # The number of domain expert
        self.scoring_func = 'softmax'
        self.aux_loss_alpha = 0.1 # aux loss is  design for balance the data amount processed by each domain expert.
        self.input_size = 256 # Input data size for the expert
        self.output_dim=256 # output data size for the expert
        self.hidden_dim=512 # hidden data size for the expert
        self.shared_expert_num=1 #The number of shared expert
```
These are parameters for predicting the solubility of protein,You can also find the parameters for predicting The effect of protein mutations on solubility 
## Train

### 1. Run ESMC Phase  
run Sol_ESMC.py or Mut_ESMC.py
Extract biological insights from the `esmc600m` model using sequence data to generate biological feature embeddings.    


### 2. Model Training Phase  
Train your model using sequence data and generated `.emb` files.You might also need a dataset for validation
Run training script (specify paths to sequence data and embeddings)  

##  Test  
Validate the trained model’s performance on a held-out dataset.  

