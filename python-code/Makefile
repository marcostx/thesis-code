#FUNCTION
define cecho
    @echo "\033[92m$(1)\033[0m"
endef

# rcnn-fer Docker Makefile
CPU_REGISTRY_URL=so77id
GPU_REGISTRY_URL=so77id
CPU_VERSION=latest
GPU_VERSION=gpu
CPU_DOCKER_IMAGE=narcostx3
GPU_DOCKER_IMAGE=narcostx3
DOCKER_USER=narcostx3
DOCKER_CONTAINER_NAME=test

# params
GPU=true
CUDA_VISIBLE_DEVICES=0

##############################################################################
############################# DOCKER VARS ####################################
##############################################################################
# COMMANDS
DOCKER_COMMAND=docker
NVIDIA_DOCKER_COMMAND=nvidia-docker

#HOST VARS
LOCALHOST_IP=127.0.0.1
HOST_TENSORBOARD_PORT=26006
HOST_NOTEBOOK_PORT=28888

#HOST CPU VARS
HOST_CPU_SOURCE_PATH=$(shell pwd)

# HOST_CPU_DATASETS_PATH=/home/marcostxs/workspace/github/datasets
HOST_CPU_DATASETS_PATH=/Users/marcostx/workspace/master/datasets
HOST_CPU_METADATA_PATH=/Users/marcostx/workspace/master/metadata

#HOST GPU PATHS
HOST_GPU_SOURCE_PATH=$(shell pwd)
#HOST_GPU_DATASETS_PATH=/home/marcostx/master-degree/masters-project-old/CCTV
# HOST_GPU_DATASETS_PATH=/home/marcostx/master-degree/CCTV
HOST_GPU_DATASETS_PATH=/datasets/sandra/media-eval-2015-violence/videos/
# HOST_GPU_DATASETS_PATH=/datasets/marcostx/rwf-2000/
HOST_GPU_METADATA_PATH=/work/$(USER)/metadata/audio

#IMAGE VARS
IMAGE_TENSORBOARD_PORT=6006
IMAGE_NOTEBOOK_PORT=8888
IMAGE_SOURCE_PATH=/home/src
IMAGE_DATASETS_PATH=/home/datasets
IMAGE_METADATA_PATH=/home/metadata
IMAGE_KINETICS_PATH=/home/datasets/kinetics

# DOCKER vars
EXP_NAME=""
HOSTNAME=$(shell cat /etc/hostname)

# VOLUMES

CPU_DOCKER_VOLUMES = --volume=$(HOST_CPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
				     --volume=$(HOST_CPU_DATASETS_PATH):$(IMAGE_DATASETS_PATH) \
				     --volume=$(HOST_CPU_METADATA_PATH):$(IMAGE_METADATA_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH) \
				     --shm-size 8G

GPU_DOCKER_VOLUMES = --volume=$(HOST_GPU_SOURCE_PATH):$(IMAGE_SOURCE_PATH) \
				     --volume=$(HOST_GPU_DATASETS_PATH):$(IMAGE_DATASETS_PATH) \
				     --volume=$(HOST_GPU_METADATA_PATH):$(IMAGE_METADATA_PATH) \
				     --workdir=$(IMAGE_SOURCE_PATH) \
				     --shm-size 8G


DOCKER_TENSORBOARD_PORTS = -p $(LOCALHOST_IP):$(HOST_TENSORBOARD_PORT):$(IMAGE_TENSORBOARD_PORT)
DOCKER_JUPYTER_PORTS = -p $(LOCALHOST_IP):$(HOST_NOTEBOOK_PORT):$(IMAGE_NOTEBOOK_PORT)

# IF GPU == false --> GPU is disabled
# IF GPU == true --> GPU is enabled
ifeq ($(GPU), true)
	DOCKER_RUN_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME) $(GPU_DOCKER_VOLUMES) $(GPU_DOCKER_IMAGE):$(GPU_VERSION)
	DOCKER_RUN_TENSORBOARD_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME) $(DOCKER_TENSORBOARD_PORTS) $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_JUPYTER_COMMAND=$(NVIDIA_DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME) $(DOCKER_JUPYTER_PORTS) $(GPU_DOCKER_VOLUMES) $(GPU_REGISTRY_URL)/$(GPU_DOCKER_IMAGE):$(GPU_VERSION)
else
	DOCKER_RUN_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME)  $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_TENSORBOARD_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME)  $(DOCKER_TENSORBOARD_PORTS) $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
	DOCKER_RUN_JUPYTER_COMMAND=$(DOCKER_COMMAND) run -it --rm --userns=host -e HOSTNAME=$(HOSTNAME) --name=$(DOCKER_CONTAINER_NAME)-$(EXP_NAME)  $(DOCKER_JUPYTER_PORTS) $(CPU_DOCKER_VOLUMES) $(CPU_REGISTRY_URL)/$(CPU_DOCKER_IMAGE):$(CPU_VERSION)
endif


# COMMANDS
JUPYTER_COMMAND=jupyter
TENSORBOARD_COMMAND=tensorboard
PYTHON_COMMAND=python3
TORCH_COMMAND=th
MKDIR_COMMAND=mkdir
WGET_COMMAND=wget
SCRIPT_LUA=util/create_t7.lua
SCRIPT_PYTHON=utils/join_h5_files.py
START=0
END=10
# URLs
C3D_URL=http://www.recod.ic.unicamp.br/~marcostx/weights/c3d.pickle

TENSORBOARD_PATH=$(IMAGE_METADATA_PATH)

setup s:
	@$(MKDIR_COMMAND) -p ./weigths
	@$(WGET_COMMAND) $(C3D_URL) -P ./weigths

run-test rtm: docker-print
	@$(DOCKER_RUN_COMMAND)

training tr:
	@$(PYTHON_COMMAND) train_main.py -dp /home/datasets

train-hockey thh:
	@$(PYTHON_COMMAND) kfold_train.py -dp /home/datasets

finetuning-efficient rwq:
	@$(PYTHON_COMMAND) train_efficient.py -dp /home/datasets

splitter sp:
	@$(PYTHON_COMMAND) utils/splitCCTV.py -dp /home/datasets -ou CCTV -a /home/datasets/annotations.json

feature-extractor fe:
		@$(PYTHON_COMMAND) utils/create_rwf_dataset.py -dp /home/datasets -s $(START) -e $(END)

hdf5-to-npy hn:
	@$(PYTHON_COMMAND) utils/hdf5_to_npy.py

sscript sk:
	@$(PYTHON_COMMAND) $(SCRIPT_PYTHON)

build-pyflow-py pf:
	$(call "[pyflow] Building pyflow")
	@$(PYTHON_COMMAND) setup.py build_ext -i

run-jupyter rj: docker-print
	@$(DOCKER_RUN_JUPYTER_COMMAND)  bash -c "make jupyter CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)"; \
	status=$$?

run-train rt: docker-print
	@$(DOCKER_RUN_COMMAND)  bash -c "make training CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)";

run-hockey rh: docker-print
	@$(DOCKER_RUN_COMMAND)  bash -c "make train-hockey CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)";

run-fine rr: docker-print
	@$(DOCKER_RUN_COMMAND)  bash -c "make finetuning-efficient CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)";

run-split r2: docker-print
	@$(DOCKER_RUN_COMMAND)  bash -c "make splitter";

run-dataset rf: docker-print
	@$(DOCKER_RUN_COMMAND)  bash -c "make feature-extractor START=$(START) END=$(END) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)";

script sq: docker-print
	@$(DOCKER_RUN_COMMAND)  bash -c "make sscript";

create-t7 d: docker-print
	@echo "[Torch] Building t7 file..."
	@$(TORCH_COMMAND) $(SCRIPT_LUA)

run-converter rc: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make hdf5-to-npy";

 build-pyflow bp: docker-print
	@$(DOCKER_RUN_COMMAND) bash -c "make build-pyflow-py";

#PRIVATE
docker-print psd:
ifeq ($(GPU), true)
	$(call cecho, "[GPU Docker] Running gpu docker image...")
else
	$(call cecho, "[CPU Docker] Running cpu docker image...")
endif
