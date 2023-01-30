import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torchvision.datasets as dset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pylab as plt
import configargparse
import os


def plot_save_figures(real_a_test,fake_b_test,real_b_test,fake_a_test,name_first,name_second, path_image):
    fig=plt.figure(figsize=(20,20))
    #Real A 
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(np.transpose(vutils.make_grid((real_a_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    ax1.axis("off")
    ax1.set_title("Real "+name_first)
    
    #Fake B
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(np.transpose(vutils.make_grid((fake_b_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    ax2.axis("off")
    ax2.set_title("Fake "+name_second)
    
    #Real B
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(np.transpose(vutils.make_grid((real_b_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    ax3.axis("off")
    ax3.set_title("Real "+name_second)
    
    #Fake A
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(np.transpose(vutils.make_grid((fake_a_test[:4]+1)/2, padding=2, normalize=True).cpu(),(1,2,0)))
    ax4.axis("off")
    ax4.set_title("Fake "+name_first)
    
    plt.savefig(path_image)

def plot_images_data(dataloader_test_first, dataloader_test_second, name_first, name_second, G_A2B, G_B2A, device, name = "image"):
    
    folder_name = './images'
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        print("Folder created")
    else:
        print("Folder already exists")
    path_image = './images/' + name + ".png"

    batch_a_test = next(iter(dataloader_test_first))[0].to(device)
    real_a_test = batch_a_test.cpu().detach()
    fake_b_test = G_A2B(batch_a_test ).cpu().detach()        

    batch_b_test = next(iter(dataloader_test_second))[0].to(device)
    real_b_test = batch_b_test.cpu().detach()
    fake_a_test = G_B2A(batch_b_test ).cpu().detach()

    plot_save_figures(real_a_test,fake_b_test,real_b_test,fake_a_test,name_first,name_second,path_image)
    
def dataloader(path_data, image_size, workers, batch_size):
    dataset = dset.ImageFolder(root=path_data,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0, 0, 0), (1, 1, 1)),
                                ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)   
    return dataloader

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--name', type=str,
                    help='name project')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number epochs')
    parser.add_argument('--config', is_config_file=True,
                    help='config file path')
    parser.add_argument("--data_train_first", type=str,
                        default='./data/simple', help='input data directory')
    parser.add_argument("--path_model", type=str,
                        default='./models/', help='directory for models')
    parser.add_argument("--data_train_second", type=str,
                        default='./data/simple', help='input data directory')
    parser.add_argument("--data_test_first", type=str,
                        default='./data/simple', help='input data directory')
    parser.add_argument("--data_test_second", type=str,
                        default='./data/simple', help='input data directory')
    parser.add_argument("--w_size", type=int, default=256,
                        help='resize image width')
    parser.add_argument("--h_size", type=int, default=256,
                        help='resize image height')
    parser.add_argument("--batch_size", type=int, default=5,
                        help='batch size')
    parser.add_argument("--workers", type=int, default=2,
                        help='how many sub-processes to use for data loading')
    return parser