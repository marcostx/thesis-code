Evaluating Attention-Based Models for Violence Classification in Videos
------------------------------------------------------------
**Marcos Teixeira**, Jan 2020

This repository contains source code for my Master's thesis, which describes a study about Attention Mechanisms in the context of Violence Classification in videos. We conducted all experiments at the [RECOD laboratory](https://recodbr.wordpress.com). The computer is equipped with a Ubuntu 16.04 operating system machine, Intel Core i7-3770K 3.50 GHz, and 128~GB of memory. For graphic-intense applications, the computer is also equipped with two NVIDIA Titan XP GPU with 12 GB of GPU memory and 3,840 CUDA cores. 

For details about the method see [PDF with the Master's thesis](https://drive.google.com/file/d/1SfwYSlannkKXnHxuOlJodt2baYoow6-g/view?usp=sharing).


### Structure
The experiments were conducted using two different programming languages. The code for [TAGM model](https://openaccess.thecvf.com/content_cvpr_2017/html/Pei_Temporal_Attention-Gated_Model_CVPR_2017_paper.html) is written in Lua-Torch (**lua-code/** folder). Additionally, the remaining models were implemented using PyTorch & Keras (**python-code/**).

The instructions to run the experiments are included in each folder.