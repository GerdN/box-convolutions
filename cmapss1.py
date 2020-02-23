# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 07:36:59 2020

@author: Nollmann
"""
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from os import path
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from box_convolution import BoxConv2d



def sliding_window(data, N_tw = 30, stride = 1):
    N_en = np.unique(data[:,0]).shape[0]                            # Number of engines (N_en)
    m = 0
    for i in range(N_en):
        n_H   = data[data[:,0] == i+1,0].shape[0]
        N_sw  = int((n_H- N_tw) / stride + 1)                       # Number of sliding windows for engine 'i' 
        for h in range(N_sw):
            m = m + 1    
    return m, N_en   

def load_dataset(N_tw, stride, sel, R_early):
    # Load training data
    basepath        = path.dirname(os.getcwd()) 
    train_set       = np.loadtxt(path.abspath(path.join(basepath, "data", "train_FD001.txt")))  # Training set
    train_set_x_org = train_set[:,sel]                              # Training set input space (x)    
    train_set_c     = train_set[:,np.array([1])]                    # Training set cycles (c)
    
    # Normalize the data
    ub = train_set_x_org.max(0)
    lb = train_set_x_org.min(0)    
    train_set_x = 2 * (train_set_x_org - lb) / (ub - lb) - 1   
   
    N_ft    = sel.shape[0]                                           # Nunber of features (N_ft)
    m, N_en = sliding_window(train_set, N_tw, stride)                # Number of training data & engines
    
    train_x = np.empty((m, N_tw, N_ft, 1), float)
    train_y = np.empty((m, 1), float)
    
    k = 0
    for i in range(N_en):
        idx       = train_set[:,0] == i+1                            # Index for engine number 'i'
        train_i_x = train_set_x[idx,:]                               # Engine 'i' training  data
        train_i_c = train_set_c[idx]                                 # Engine 'i' cycles (c)
        train_i_y = train_i_c[-1] - train_i_c                        # RUL: Remaining Useful Lifetime for engine 'i'
        train_i_y[train_i_y > R_early] = R_early                     # R_early = 125
        N_sw      = int((train_i_x.shape[0] - N_tw) / stride + 1)    # Number of sliding windows for engine 'i' 
        for h in range(N_sw):
            k = k + 1
            vert_start = h * stride
            vert_end   = h * stride + N_tw
            train_i_x_slice = train_i_x[vert_start:vert_end,:]       # Training input data for engine 'i' on time window 'h'
            train_i_y_slice = train_i_y[vert_end-1,:]                # Training output data for engine 'i' on time window 'h'
            train_i_x_slice.shape = (N_tw, N_ft, 1)                  # Reshape training set input (N_tw, N_ft, 1)
            train_i_y_slice.shape = (1, 1)                           # Reshape training set output (1, 1)
            train_x[k-1,:,:] = train_i_x_slice
            train_y[k-1,:] = train_i_y_slice
     
    # Load test data
    test_set       = np.loadtxt(path.abspath(path.join(basepath, "data", "test_FD001.txt")))
    test_set_x_org = test_set[:,sel]                                 # Test set input space (x)
    test_set_c     = test_set[:,np.array([1])]                       # Test set cycles (c)
    test_y         = np.loadtxt(path.abspath(path.join(basepath, "data", "RUL_FD001.txt")))    # Test set RUL (c)
    test_y.shape   = (test_y.shape[0], 1)
    
    # Normalize the data
    test_set_x = 2 * (test_set_x_org - lb) / (ub - lb) - 1   
    
    m_ts, N_en_ts = sliding_window(test_set, N_tw, stride)           # Number of training data & engines
    
    test_x = np.empty((N_en_ts, N_tw, N_ft, 1), float)
    
    k = 0
    for ii in range(N_en_ts):
        engine         = test_set[:,0] == ii+1                       # Index for engine number 'i'
        test_i_x       = test_set_x[engine,:]                        # Engine 'i' test  data
        test_i_x_slice = test_i_x[-N_tw:,:]                          # Training input data for engine 'i' on time window 'h'
        test_i_x_slice.shape = (N_tw, N_ft, 1)                       # Reshape training set input (N_tw, N_ft, 1)
        test_x[ii,:,:] = test_i_x_slice
    
    return train_x, train_y, test_x, test_y

N_tw     = 30                                                               # Time Window (N_tw)
R_early  = 125                                                              # Max RUL in training set
stride   = 1
sel      = np.array([6, 7, 8, 11, 12, 13, 15, 16, 17, 18, 19, 21, 24, 25])  # Index of input features

X_train, Y_train, X_test, Y_test = load_dataset(N_tw, stride, sel, R_early)
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
# and then we send them to the chosen device
X_train_tensor = torch.from_numpy(X_train).float().to(device)
Y_train_tensor = torch.from_numpy(Y_train).float().to(device)
X_test_tensor = torch.from_numpy(Y_test).float().to(device)
Y_test_tensor = torch.from_numpy(Y_test).float().to(device)
# Here we can see the difference - notice that .type() is more useful
# since it also tells us WHERE the tensor is (device)
print(type(X_train), type(X_train_tensor), X_train_tensor.type())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = BoxConv2d(1, 40, 28, 28)
        self.conv1_1x1 = nn.Conv2d(40, 40, 1, 1)

        self.fc1 = nn.Linear(7*7*40, 10)

    def forward(self, x):
        # The following line computes responses to 40 "generalized Haar filters"
        x = self.conv1_1x1(self.conv1(x))
        x = F.relu(F.max_pool2d(x, 4))

        x = self.fc1(x.view(-1, 7*7*40))
        return F.log_softmax(x, dim=1)

try:
    import cv2
    box_video_resolution = (300, 300)
    box_video = cv2.VideoWriter(
        'mnist-boxes.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, tuple(reversed(box_video_resolution)))
    box_video_frame_count = 0
    video_background = None # to be defined in `main()`, sorry for globals and messy code
except ImportError:
    box_video = None
    print('Couldn\'t import OpenCV. Will not log boxes to a video file')


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # log boxes to a video file
        if box_video is not None:
            global box_video_frame_count
            
            # change video background
            if box_video_frame_count % 5 == 0:
                global video_background # defined at the top for beautiful box visualization
                sample_idx = torch.randint(len(train_loader.dataset), (1,)).item()
                sample_digit = train_loader.dataset[sample_idx][0]
                video_background = torch.nn.functional.pad(sample_digit, (14,14,14,14))
                video_background = torch.nn.functional.interpolate(
                    video_background.unsqueeze(0), size=box_video_resolution, mode='nearest')[0,0]
                video_background = video_background.unsqueeze(-1).repeat(1, 1, 3)
                video_background = video_background.mul(255).round().byte().numpy()

            # log boxes to the video file
            if batch_idx % 5 == 0:
                box_importances = model.conv1_1x1.weight.detach().float().abs().max(0)[0].squeeze()
                box_importances /= box_importances.max()
                boxes_plot = model.conv1.draw_boxes(
                    resolution=box_video_resolution, weights=box_importances)
                box_video.write(cv2.addWeighted(boxes_plot, 1.0, video_background, 0.25, 0.0))
                box_video_frame_count += 1

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.MSELoss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        for g in optimizer.param_groups:
            g['lr'] *= 0.999

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    use_cuda = torch.cuda.is_available()
    batch_size = 64
    n_epochs = 10

    torch.manual_seed(666)

    device = torch.device('cuda' if use_cuda else 'cpu')

    #cmapss_train = df_train(
        #'./', train=True, download=False, transform=transforms.ToTensor())
     #   './', train=True, download=False)#, transform=transforms.ToTensor())
    #cmapss_test = df_test(
     #   './', train=False)#, transform=transforms.ToTensor())
    #dataset = TensorDataset(X_train_tensor, Y_train_tensor) 
    #train_loader = DataLoader(dataset=train_dataset, batch_size=16)
    #val_loader = DataLoader(dataset=val_dataset, batch_size=20)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        X_train_tensor, Y_train_tensor, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        X_test_tensor,  Y_test_tensor, batch_size=batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, n_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        
if __name__ == '__main__':
    main()
