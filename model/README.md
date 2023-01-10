This repository contains the code for the QSAR models for Rat Liver Microsomal Stability endpoints. To use the application locally, you can either use [Git](https://git-scm.com/) to clone the respository, or you can simply download a ZIP file (by clicking the green "Code" button on the top right corner) and then unzip it. The next steps are described below.

If you use Git to clone this repository, please use the --recursive flag to ensure the chemprop submodule is updated:

`https://github.com/ersilia-os/eos5505.git`

## Installing required software

1. Install [anaconda or miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html#)

Python is also required but it is included with either installation of conda or miniconda.

## Setting up the environment

You only have complete these steps one time.

1. Open your terminal
  - If you're on Windows, open Anaconda Prompt (window -> Anaconda3 -> Anaconda Prompt)
  - If you're on Mac or Linux, open your Terminal
2. Create environment
  - For Windows and Linux machines
    1. Type `conda env create --prefix ./env -f environment.yml` and hit Enter
    2. Wait several minutes for the envitonment to be created
    3. For Windows machines only, type `pip install typed-argument-parser` and hit Enter
  - For Mac machines
    1. Type `conda env create --prefix ./env -f environment_mac.yml` and hit Enter
    2. Wait several minutes for the envitonment to be created
    
## Running the application

1. If you're doing this immediately after completing the steps above, skip to step 4
2. Open your terminal
  - If you're on Windows, open Anaconda Prompt (window -> Anaconda3 -> Anaconda Prompt)
  - If you're on Mac or Linux, open your terminal
3. Type `conda activate ./env` and hit Enter
4. Type `python app.py` and hit Enter
5. Open Chrome or Firefox and browse to `http://127.0.0.1:5000/`
6. To close the application, hit `Ctrl + c` or `Cmd + c` in the Terminal and then type `conda deactivate` and hit Enter to close the conda environment