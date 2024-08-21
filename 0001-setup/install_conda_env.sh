# Install Nornour conda environment 

. ~/conda_init.sh

conda create --name Nornour
conda activate Nornour

#### MACE

# (optional) Install MACE's dependencies from Conda as well
conda config --add channels conda-forge
conda config --set channel_priority strict

conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy scipy matplotlib ase opt_einsum prettytable pandas e3nn
python -m pip install wandb
python -m pip install h5py

# Clone and install MACE (and all required packages)
git clone -b foundations https://github.com/ACEsuit/mace.git
pip install ./mace

pip install netCDF4 # for converting trajectories

#### MACE END

conda install conda-forge::gromacs

# Martini CG membrane builder
pip install insane

# CGSB/COBY - Python mebrane builder (Coarse-grained System Builder)
pip install COBY

# Best backmapping tool? ezAlign? Backward

