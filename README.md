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
dataset/
    apple/
        apple1.jpg
        apple2.jpg
        apple3.jpg
        ...
    banana/
        banana1.jpg
        banana2.jpg
        banana3.jpg
        ...
```

Once you have your datasets prepared, you can run the training script to begin training a CycleGAN model:

```
python train_sgan.py --config config_ab.txt
```

You can also specify other optional arguments to customize the training process. For a full list of available arguments, see the `load_data.py` file.

## Sample

The following are the input parameters for the parser: 
--name: a string that serves as the name of the project.
--num_epochs: an integer that specifies the number of epochs the model should run for.
--config: a boolean that indicates the configuration file path.
--data_train_first: a string that specifies the path to the first training dataset.
--path_model: a string that specifies the directory for the models.
--data_train_second: a string that specifies the path to the second training dataset.
--data_test_first: a string that specifies the path to the first test dataset.
--data_test_second: a string that specifies the path to the second test dataset.
--w_size: an integer that specifies the desired width for the images.
--h_size: an integer that specifies the desired height for the images.
--batch_size: an integer that specifies the batch size for training.
--workers: an integer that specifies the number of sub-processes to use for data loading.

This is the description of the arguments used in the parser:
```
--name : this is the name of the project

--num_epochs : this is the number of epochs used for training the model

--config : this is the configuration file path

--data_train_first : this is the input data directory for the first training dataset

--path_model : this is the directory for the models

--data_train_second : this is the input data directory for the second training dataset

--data_test_first : this is the input data directory for the first test dataset

--data_test_second : this is the input data directory for the second test dataset

--w_size : this is the resize image width

--h_size : this is the resize image height

--batch_size : this is the batch size used for training and testing

--workers : this is the number of sub-processes used for data loading
```

## Results

The results of the CycleGAN training process can be found in the `results/` directory. This directory contains two subdirectories, `A2B/` and `B2A/`, that contain the generated images from each domain.

## Acknowledgments

This implementation of CycleGAN is based on the work by [Jun-Yan Zhu et al.](https://arxiv.org/abs/1703.10593). This repository is maintained by [@username](https://github.com/username).