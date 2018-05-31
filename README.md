
# Synaptic Clefts Detection in EM images

This repository is a re-implementation of [Synapse-unet](https://github.com/zudi-lin/synapse-unet) for synaptic cleft detection from electron microscopy (EM) images using PyTorch. However, it contains some enhancements:

* Add residual blocks to orginal unet.
* Change concatenation to summation.

----------------------------

## Dataset

## Training

### Command
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u train.py -t /cremi/images/ -dn im_A.h5@im_B.h5@im_C.h5 -ln syn_A.h5@syn_B.h5@syn_C.h5 -o outputs -lr 0.001 --volume-total 100000 --volume-save 10000 -mi 24,256,256 -g 4 -c 6 -b 4
```
### Visulazation
* Visualize the training loss using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).
* Use TensorBoard with `tensorboard --logdir runs`  (needs to install TensorFlow).
## Prediction

## Evaluation

## TODO

* Add augmentation.
* Add auxiliary boundary detection.

