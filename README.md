---
title: README
author: Nathan QUIBLIER
---


# Synopsis
MLFCS is a framework to infer physical parameters of particles (model of motion,anomalous coefficient, diffusion coefficient). It is based on the analysis of confocal fluorescence correlation spectroscopy (FCS) data using machine learning methods.

MLFCS is a naive model that can be trained with any model of motion of interest and in any parameter range of interest.

You can also load you own trained model ('data/model_trained').

A medium-size test set is provided to test the algorithm performance('data/test_set_git').


# Installation

MLFCS can be installed from source with any standard Python package manager that supports [pyproject.toml](pyproject.toml) files. For example, to install it with pip, either locally or in a virtual environment, run the following commands:

~~~sh
git clone https://gitlab.inria.fr/nquilbie/mlfcs
cd MLFCS
# Uncomment the following 2 lines to create and activate a virtual environment.
# python -m venv myvenv
# source venv/bin/activate (if Linux Mac)
# source venv/Scripts/activate (if windows)
python -m pip install .
pip install ipykernel
~~~


# Usage

Two notebooks are included to illustrate how MLFCS can be used (select the kernel from your virtual environment to run the notebook):
* src/Test.ipynb computes figure 1, 2 and 3 of the paper on the provided test set (you only have to run all).
* src/Trainer.ipynb trains and tests the model on a train and test set provided by the user
