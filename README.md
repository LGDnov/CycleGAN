# CycleGAN

This repository contains implementations of CycleGAN, a method for learning image-to-image translation without paired data. CycleGAN was proposed by [Zhu et al.](https://arxiv.org/abs/1703.10593) and is a type of generative adversarial network (GAN) that is able to learn mappings between two image domains.

## Prerequisites

This implementation of CycleGAN requires Python 3.6 and a GPU. The following libraries and packages are also required:

- TensorFlow 2.x
- NumPy
- Matplotlib
- SciPy

## Usage

To train a CycleGAN model, you need to provide two datasets of images. Each dataset should be in its own folder and should contain only images of the same domain.

Example:

```
data/
    horses/
        horse1.jpg
        horse2.jpg
        horse3.jpg
        ...
    zebras/
        zebra1.jpg
        zebra2.jpg
        zebra3.jpg
        ...
```

Once you have your datasets prepared, you can run the training script to begin training a CycleGAN model:

```
python train.py --data_dir data
```

You can also specify other optional arguments to customize the training process. For a full list of available arguments, see the `train.py` file.

## Results

The results of the CycleGAN training process can be found in the `results/` directory. This directory contains two subdirectories, `A2B/` and `B2A/`, that contain the generated images from each domain.

## Acknowledgments

This implementation of CycleGAN is based on the work by [Jun-Yan Zhu et al.](https://arxiv.org/abs/1703.10593). This repository is maintained by [@username](https://github.com/username).