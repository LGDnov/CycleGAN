# CycleGAN

This repository contains implementations of CycleGAN, a method for learning image-to-image translation without paired data. CycleGAN was proposed by [Zhu et al.](https://arxiv.org/abs/1703.10593) and is a type of generative adversarial network (GAN) that is able to learn mappings between two image domains.

## Prerequisites

This implementation of CycleGAN requires Python 3.9 and a PyTorch. The following libraries and packages are also required:

- PyTorch
- NumPy
- Matplotlib

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

## Parameters for network training

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

--load_model_path: Specifies the path to load a pre-trained model.

--load_model_name: Specifies the name of the pre-trained model to be loaded.
```

These parameters are set in the `config_ab.txt` file.

## Results

The results of the CycleGAN training process can be found in the `images/` directory. This directory will contain images obtained during network training.

### Apple

![Gift](https://github.com/LGDnov/CycleGAN/blob/main/gif/apple_gif.gif)

### Banana

![Gift](https://github.com/LGDnov/CycleGAN/blob/main/gif/banana_gif.gif)


## Acknowledgments

This implementation of CycleGAN is based on the work by [Jun-Yan Zhu et al.](https://arxiv.org/abs/1703.10593). This repository is maintained by [@LGDnov](https://github.com/LGDnov).