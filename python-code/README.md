Evaluating Attention-Based Models for Violence Classification in Videos (Python)
------------------------------------------------------------
**Marcos Teixeira**, Fev 2020

## Requirements
* [Make](https://www.gnu.org/software/make/)
* [Docker](https://www.docker.com)

test
## Parameters
* GPU=`true|false`
* CUDA_VISIBLE_DEVICES=`GPU_ID`

## Build the Docker image 
Note that this will take an hour or two depending on your machine since it compiles a few libraries from scratch.

* docker build -t machine:gpu -f Dockerfile.pytorch .

## Commands

## extract frames
* `make run-dataset GPU=true CUDA_VISIBLE_DEVICES=0`

## train
* `make run-train GPU=true CUDA_VISIBLE_DEVICES=0`


@Author