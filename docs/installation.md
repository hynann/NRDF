# Getting Started

The code was tested under `Ubuntu 22.04, Python 3.9, CUDA 11.3, PyTorch 1.12.1`.\
Follow the steps below to create a conda environment with necessary dependencies:

## 1. Create environment
    conda create -n nrdf python=3.9
    conda activate nrdf


## 2. Install pytorch
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


## 3. Install other dependencies
    pip install -r requirements.txt
  
## 4. Install faiss-gpu
    conda install -c pytorch faiss-gpu


