
# Synaptic Clefts Detection in EM images

This repository is a re-implementation of [Synapse-unet](https://github.com/zudi-lin/synapse-unet) (in Keras) for synaptic clefts detection in electron microscopy (EM) images using PyTorch. However, it contains some enhancements of the original model:

* Add residual blocks to orginal unet.
* Change concatenation to summation in the expansion path.

----------------------------

## Installation

* Clone this repository : `git clone --recursive https://github.com/zudi-lin/synapse_pytorch.git`
* Create a conda environment :  `conda env create -f synapse_pytorch/envs/py3_pytorch.yml`

## Dataset

Training and testing data comes from MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images ([CREMI challenge](https://cremi.org)). Three training volumes of adult *Drosophila melanogaster* brain imaged with serial section Transmission Electron Microscopy (ssTEM) are provided.

## Training

### Command

* Activate previously created conda environment : `source activate ins-seg-pytorch`.
* Run `train.py`.

```
usage: train.py [-h] [-t TRAIN] [-dn IMG_NAME] [-ln SEG_NAME] [-o OUTPUT]
                [-mi MODEL_INPUT] [-ft FINETUNE] [-pm PRE_MODEL] [-lr LR]
                [--volume-total VOLUME_TOTAL] [--volume-save VOLUME_SAVE]
                [-g NUM_GPU] [-c NUM_CPU] [-b BATCH_SIZE]

Training Synapse Detection Model

optional arguments:
  -h, --help                Show this help message and exit
  -t, --train               Input folder
  -dn, --img-name           Image data path
  -ln, --seg-name           Ground-truth label path
  -o, --output              Output path
  -mi, --model-input        I/O size of deep network
  -ft, --finetune           Fine-tune on previous model [Default: False]
  -pm, --pre-model          Pre-trained model path
  -lr                       Learning rate [Default: 0.0001]
  --volume-total            Total number of iterations
  --volume-save             Number of iterations to save
  -g, --num-gpu             Number of GPUs
  -c, --num-cpu             Number of CPUs
  -b, --batch-size          Batch size
```

The script supports training on datasets from multiple directories. Please make sure that the input dimension is in *zyx*.

### Visulazation
* Visualize the training loss using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).
* Use TensorBoard with `tensorboard --logdir runs`  (needs to install TensorFlow).

## Prediction

* Run `test.py`.

```
usage: test.py  [-h] [-t TRAIN] [-dn IMG_NAME] [-o OUTPUT] [-mi MODEL_INPUT]
                [-g NUM_GPU] [-c NUM_CPU] [-b BATCH_SIZE] [-m MODEL]

Testing Synapse Detection Model

optional arguments:
  -h, --help                Show this help message and exit
  -t, --train               Input folder
  -dn, --img-name           Image data path
  -o, --output              Output path
  -mi, --model-input        I/O size of deep network
  -g, --num-gpu             Number of GPUs
  -c, --num-cpu             Number of CPUs
  -b, --batch-size          Batch size
  -m, --model               Model path used for test
```

## Evaluation

Run `evaluation.py -p PREDICTION -g GROUND_TRUTH`.
The evaluation script will count the number of false positive and false negative pixels based on the evaluation metric from [CREMI challenge](https://cremi.org/metrics/). Synaptic clefts IDs are NOT considered in the evaluation matric. The inputs will be converted to binary masks.

## TODO

* Use ELU activation.
* Add augmentation.
* Add auxiliary boundary detection.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/zudi-lin/synapse_pytorch/blob/master/LICENSE) file for details.
