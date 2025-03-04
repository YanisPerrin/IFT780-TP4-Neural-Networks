#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License: Opensource, free to use
Other: Suggestions are welcome
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from manage.CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from manage.HDF5Dataset import HDF5Dataset
from models.AlexNet import AlexNet
from models.CNNVanilla import CnnVanilla
from models.ResNet import ResNet
from models.UNet import UNet
from models.VggNet import VggNet
from models.yourSegNet import YourSegNet
from models.yourUNet import YourUNet
from torchvision import datasets

from loss.DiceLoss import DiceLoss
from loss.AsymLoss import AsymLoss
from loss.DiceCELoss import DiceCELoss


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model UNet [hyper_parameters]'
                                           '\n python3 train.py --model UNet --predict',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets. Be aware that when using UNet model there is no"
                                                 " need to provide a dataset since UNet model only train "
                                                 "on acdc dataset.",
                                     add_help=True)
    parser.add_argument('--model', type=str, default="CnnVanilla",
                        choices=["CnnVanilla", "VggNet", "AlexNet", "ResNet", "yourUNet", "yourSegNet", "UNet"])
    parser.add_argument('--dataset', type=str, default="acdc", choices=["cifar10", "svhn", "acdc"])
    parser.add_argument('--loss', type=str, default="CE", choices=["CE", "Dice", "DiceCE", "Asym"])
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data_aug', type=int, default=None, choices=[0, 1, 2],
                        help="Data augmentation")
    parser.add_argument('--predict', type=str,
                        help="Name of the file containing model weights used to make "
                             "segmentation prediction on test data")
    parser.add_argument('--save_checkpoint', action='store_true',
                        help="Save the model's weights after training in 'checkpoints' folder")
    parser.add_argument('--load_checkpoint', type=str,
                        help="Load the model's weights from the specified file")
    return parser.parse_args()


if __name__ == "__main__":

    args = argument_parser()

    loss = args.loss
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    data_augment = args.data_aug
    save_checkpoint = args.save_checkpoint
    load_checkpoint = args.load_checkpoint

    if save_checkpoint:
        print('Checkpoints will be saved!')
    else:
        print('Checkpoints will NOT be saved!')

    if load_checkpoint is not None:
        if Path(load_checkpoint).exists():
            print('Checkpoint file found!')
        else:
            raise Exception("Checkpoint file not found!")

    if data_augment is not None:
        print(f'Data augmentation {data_augment} activated!')
    else:
        print('Data augmentation NOT activated!')

    # set hdf5 path according your hdf5 file location
    acdc_hdf5_file = '../data/ift780_acdc.hdf5'

    # Set transformations. To see the transformations, check data_augmentation.ipynb.
    base_transform = transforms.Compose([transforms.ToTensor()])
    if data_augment == 0:
        transform = transforms.Compose([
            base_transform,
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.2)
        ])
    elif data_augment == 1:
        transform = transforms.Compose([
            base_transform,
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(degrees=15)
        ])
    elif data_augment == 2:
        transform = transforms.Compose([
            base_transform,
            transforms.RandomRotation(degrees=15),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.2)
        ])
    else:
        transform = base_transform

    if args.dataset == 'cifar10':
        # Download the train and test set and apply transform on it
        train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=base_transform)
    elif args.dataset == 'svhn':
        # Download the train and test set and apply transform on it
        train_set = datasets.SVHN(root='../data', split='train', download=True, transform=transform)
        test_set = datasets.SVHN(root='../data', split='test', download=True, transform=base_transform)
    elif args.dataset == 'acdc':
        train_set = HDF5Dataset('train', acdc_hdf5_file, transform=transform)
        test_set = HDF5Dataset('test', acdc_hdf5_file, transform=base_transform)
    else:
        raise Exception("Dataset not found!")

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(torch.optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    if args.model == 'CnnVanilla':
        model = CnnVanilla(num_classes=10)
    elif args.model == 'AlexNet':
        model = AlexNet(num_classes=10)
    elif args.model == 'VggNet':
        model = VggNet(num_classes=10)
    elif args.model == 'ResNet':
        model = ResNet(num_classes=10)
    elif args.model == 'yourSegNet':
        model = YourSegNet(num_classes=4)
    elif args.model == 'yourUNet':
        model = YourUNet(num_classes=4)
    elif args.model == 'UNet':
        model = UNet(num_classes=4)

    if loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    elif loss == 'Dice':
        loss_fn = DiceLoss()
    elif loss == 'DiceCE':
        loss_fn = DiceCELoss()
    elif loss == 'Asym':
        loss_fn = AsymLoss()
    else:
        raise Exception("Loss function not found!")

    model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        batch_size=batch_size,
                                        loss_fn=loss_fn,
                                        optimizer_factory=optimizer_factory,
                                        validation=val_set,
                                        use_cuda=True,
                                        save_checkpoint=save_checkpoint,
                                        load_checkpoint=load_checkpoint)

    if args.predict is not None:
        model.load_weights(args.predict)
        print("predicting the mask of a randomly selected image from test set")
        model_trainer.plot_image_mask_prediction()
    else:
        print("Training {} on {} for {} epochs".format(model.__class__.__name__, args.dataset, args.num_epochs))
        model_trainer.train(num_epochs)
        model_trainer.evaluate_on_test_set()
        model_trainer.plot_metrics()
