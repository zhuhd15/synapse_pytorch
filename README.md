
# Synaptic Clefts Detection in EM images

This repository is a re-implementation of [Synapse-unet](https://github.com/zudi-lin/synapse-unet) for synaptic cleft detection from electron microscopy (EM) images using PyTorch. However, it contains some enhancements of the original model:

* Add residual blocks to orginal unet.
* Change concatenation to summation.

----------------------------

## Installation

* Clone this repository : 'git clone --recursive https://github.com/zudi-lin/synapse_pytorch.git'
* Create a conda environment :  `conda env create -f synapse_pytorch/envs/py3_pytorch.yml'
## Dataset

Training and testing data comes from MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images ([CREMI challenge](https://cremi.org)). Three training volumes of adult *Drosophila melanogaster* brain imaged with serial section Transmission Electron Microscopy (ssTEM) are provided.

## Training

### Command

* Activate previously created conda environment : `source activate ins-seg-pytorch`.
* Run 'train.py'.

```
usage: train.py [-h] [-t TRAIN] [-dn IMG_NAME] [-ln SEG_NAME] [-o OUTPUT]
                [-mi MODEL_INPUT] [-ft FINETUNE] [-pm PRE_MODEL] [-lr LR]
                [--volume-total VOLUME_TOTAL] [--volume-save VOLUME_SAVE]
                [-g NUM_GPU] [-c NUM_CPU] [-b BATCH_SIZE]

Training Synapse Detection Model

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        Input folder (train)
  -dn IMG_NAME, --img-name IMG_NAME
                        Image data path
  -ln SEG_NAME, --seg-name SEG_NAME
                        Ground-truth label path
  -o OUTPUT, --output OUTPUT
                        Output path
  -mi MODEL_INPUT, --model-input MODEL_INPUT
                        I/O size of deep network
  -ft FINETUNE, --finetune FINETUNE
                        Fine-tune on previous model [Default: False]
  -pm PRE_MODEL, --pre-model PRE_MODEL
                        Pre-trained model path
  -lr LR                Learning rate
  --volume-total VOLUME_TOTAL
                        Total number of iteration
  --volume-save VOLUME_SAVE
                        Number of iteration to save
  -g NUM_GPU, --num-gpu NUM_GPU
                        Number of gpu
  -c NUM_CPU, --num-cpu NUM_CPU
                        Number of cpu
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
```
### Visulazation
* Visualize the training loss using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).
* Use TensorBoard with `tensorboard --logdir runs`  (needs to install TensorFlow).
## Prediction

## Evaluation

## TODO

* Use ELU activation.
* Add augmentation.
* Add auxiliary boundary detection.

