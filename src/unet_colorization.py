"""
Colorization UNet: Converting greyscale images to RGB images through color classification.

Author :                    Arnold Yeung
Date :                      February 2019

Updates:
2019-09-11          (AY)            Modified code for publication.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import numpy as np
from datetime import datetime
import math
import matplotlib.pyplot as plt
import PIL

from color_classification import evaluate
from utils import *

class UNet(nn.Module):
    
    def __init__(self, params, num_colors, loss_fn=None):
        """
        Arguments:
            params{}:
                kernel (int) :              kernel size
                num_filters (int) :         number of filters 
                num_in_channels (int) :     width or height of image
            num_colors (int) :              number of colors to choose from
            loss_fn (fcn) :                 function used to calculate loss
        """
        super(UNet, self).__init__()
        
        self.kernel_size = params['kernel_size']
        self.num_filters = params['num_filters']
        self.num_colors = num_colors
        self.in_channels = params['num_in_channels']
        self.padding = self.kernel_size // 2
        self.num_layers = 2
        self.loss_fn = loss_fn
        
        #   downsample layers
        self.down_conv_layers = nn.ModuleList().cuda()                  #   list
        in_channels = self.in_channels
        out_channels = 32
        num_filters = self.num_filters
        for layer_idx in range(self.num_layers):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, 
                                                   self.kernel_size, padding=self.padding))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            self.down_conv_layers.append(nn.Sequential(*layers).cuda())
            
            in_channels = out_channels
            out_channels *= 2
            num_filters *= 2

        #   refactor layer
        self.refactor_layers = nn.ModuleList().cuda()
        out_channels = in_channels
        num_filters = int(num_filters/2)
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, self.kernel_size,
                                             padding=self.padding))
        layers.append(nn.BatchNorm2d(num_filters))
        layers.append(nn.ReLU())
        self.refactor_layers.append(nn.Sequential(*layers).cuda())

        #   upsample layers
        self.up_conv_layers = nn.ModuleList().cuda()
        in_channels *= 2
        out_channels = int(out_channels/2)
        num_filters = int(num_filters/2)
        for layer_idx in range(self.num_layers):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, 
                                                 self.kernel_size, padding=self.padding))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.Upsample(scale_factor=2))
            self.up_conv_layers.append(nn.Sequential(*layers).cuda())
            
            in_channels = out_channels * 2
            if layer_idx >= self.num_layers-2:      #   change for final layer
                out_channels = 3
                num_filters = 3
            else:
                out_channels *= 2
                num_filters *= 2
        
        #   final layer
        in_channels = out_channels
        self.final_layers = nn.ModuleList().cuda()
        layers = []
        layers.append(nn.Conv2d(self.in_channels + in_channels, self.num_colors, self.kernel_size, 
                                           padding=self.padding))
        self.final_layers.append(nn.Sequential(*layers).cuda())

    def forward(self, X):
        """
        Forward pass.
        Arguments:
            X (np.array) :      input array (num_samples * height * width * 3)
        Returns:
            Output array
        """
                
        out1 = self.down_conv_layers[0](X)
        out2 = self.down_conv_layers[1](out1)
        out3 = self.refactor_layers[0](out2)
        out4 = self.up_conv_layers[0](torch.cat((out2, out3), 1))
        out5 = self.up_conv_layers[1](torch.cat((out1, out4), 1))
        out = self.final_layers[0](torch.cat((X, out5), 1))
        
        """
        layer_lists = [self.down_conv_layers, self.refactor_layers, 
                       self.up_conv_layers, self.final_layers]
        out = X
        for layers in layer_lists:
            for layer in layers:
                out = layer(out)
        """
        return out
    
    def compute_loss(self, outputs, labels):
        """
        Compute the loss.
        Arguments:
            outputs (torch.tensor) :        output of this model (num_samples * num_colors * height * width)
            labels (torch.tensor) :         desired output of this model (num_samples * 1 * height * width)
        """
        
        #   convert inputs to one-hot (num_samples * height * width, num_colors)
        
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        
        num_samples = outputs.shape[0]
        width = outputs.shape[2]
        height = outputs.shape[3]
        
        #   reshape to (num_samples * height * width * num_colors/1)
        outputs = outputs.transpose(1, 3).contiguous()
        labels = labels.transpose(1, 3).contiguous()
        
        #   combine all samples and dimensions
        outputs = outputs.view((num_samples * height * width, self.num_colors))
        
        labels = labels.view((num_samples * height * width, ))        
        
        return self.loss_fn(outputs, labels)
        
    
def train(params, colors, model=None, report_dir=None):
    
    #   parameters
    num_epochs = params['num_epochs']
    learning_rate = params['learning_rate']
    learning_decay = params['learning_decay']
    batch_size = params['batch_size']
    report_name = params['report_file']
    report_every_epoch = params['report_every_epoch']
    report_every_batch = params['report_every_batch']
    data_dir = params['download_dir']
    batch_size = params['batch_size']
    image_classes = params['image_classes']
    num_colors = np.shape(colors)[0]
    
    #   prepare report files
    overview_filename = report_name + "_overview.txt"
    
    overview_file = open(report_dir + overview_filename, "w+")
    
    #   lists containing accuracies of all epochs (for plotting)
    train_losses = []
    valid_losses = []
  
    print("Retrieving dataset...")
    training_loader, valid_loader = load_cifar10(data_dir, batch_size, labels=image_classes)
    print("Training batches: ", len(training_loader), "Validation batches: ", len(valid_loader), flush=True)
    
    #   set optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #   train cycle
    for epoch in range(0, num_epochs):
        
        print("Start Epoch: ", str(epoch), flush=True)
        
        if epoch % report_every_epoch == 0:
            #   create train epoch report
            epoch_filename = report_name + "_e" + str(epoch) + ".txt"
            epoch_file = open(report_dir + epoch_filename, "w+")
           
        train_epoch_loss = 0
        valid_epoch_loss = 0
        
        model.train()
        
        #   for every batch
        for i, batches in enumerate(training_loader):
            
            #   get batch
            rgb_batch = batches[0].cuda()                        #   num_samples * 3 * height * width
            min_pixel = float(torch.min(rgb_batch).item())
            max_pixel = float(torch.max(rgb_batch).item())
                        
            grey_batch = rgb_to_greyscale(rgb_batch, max_pixel).cuda()      #   num_samples * 1 * height * width
            
            #   convert to color categories
            if model.in_channels == 1:
                rgb_batch = torch.tensor(map_rgb_to_categories(rgb_batch, colors)).cuda()     #  num_samples * 1 * height * width
            elif model.in_channels != 3:
                raise ValueError("Model num_in_channels is neither 1 nor 3.")
            rgb_batch = torch.tensor(rgb_batch).cuda()
            
            #   check memory size
            #print("data_and_labels is using", sys.getsizeof(data_and_labels), "bytes.")
            #print("real_data is using", sys.getsizeof(real_data), "bytes.")
            
            optimizer.zero_grad()
            
            #   generator forward pass            
            output = model(grey_batch)              #   num_samples * num_colors * height * width
            loss = model.compute_loss(output, rgb_batch)
            
            train_epoch_loss += loss.item()
            loss.backward()     #   backward pass
            optimizer.step()    #   gradient step
            
        #   calculate validation loss and accuracy
        model.eval()
        for j, batches in enumerate(valid_loader):
            
            #   get batch
            rgb_batch = batches[0].cuda()                        #   num_samples * 3 * height * width
            min_pixel = float(torch.min(rgb_batch).item())
            max_pixel = float(torch.max(rgb_batch).item())
                        
            grey_batch = rgb_to_greyscale(rgb_batch, max_pixel).cuda()      #   num_samples * 1 * height * width
            
            #   convert to color categories
            if model.in_channels == 1:
                cat_batch = torch.tensor(map_rgb_to_categories(rgb_batch, colors)).cuda()     #  num_samples * 1 * height * width
            elif model.in_channels != 3:
                raise ValueError("Model num_in_channels is neither 1 nor 3.")
            else:
                cat_batch = torch.tensor(rgb_batch).cuda()
            
            output = model(grey_batch)
            loss = model.compute_loss(output, cat_batch)
            valid_epoch_loss += loss.item()
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            #   plot for visualization (first 10 from final batch)
            max_output = output.max(dim=1)[1].unsqueeze(dim=1)                  #   num_samples * 1 * height *width
            predictions = map_categories_to_rgb(max_output[:10, :, :, :], colors).float()
            labels = map_categories_to_rgb(cat_batch[:10, :, :, :], colors).float()
            grey_images = grey_batch[:10, : , : , :]
            plot_image_set(grey_images, predictions, labels, rgb_batch, path=report_dir+'valid_e'+str(epoch)+'.png')
        
        
        #   save progress
        if (epoch+1) % 50 == 0:
            checkpoint = {'model': UNet(model_params, num_colors),
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, report_dir+'model_e'+str(epoch)+'.pth')
        
        #   average of losses
        train_epoch_loss = train_epoch_loss / i
        valid_epoch_loss = valid_epoch_loss / j
        train_losses.append(train_epoch_loss)
        valid_losses.append(valid_epoch_loss)
        
        #   save results
        print_string = "Epoch: " + str(epoch) + "   Train Loss: " + str(train_epoch_loss) + \
                        "   Valid Loss: " + str(valid_epoch_loss)
        print(datetime.now(), flush=True)
        print(print_string, flush=True)
        
        overview_file.write(print_string + "\n\n")
        if epoch % report_every_epoch == 0:
            #   write to epoch results
            epoch_file.write(print_string)
            epoch_file.close()
        
    overview_file.close()
    
    #   plot time series
    legend = ['train', 'valid']
    plot_time_series([train_losses, valid_losses], legend=legend, title='Losses', save_loc=report_dir,
                     save_name='losses', save_format='svg')
    
    return model
    

def main(model_params, train_params, train_mode=True, inference_image=""):
    """
    Main pipeline for experiment.
    Arguments:
        model_params (dict) :                   contains parameters for model
        train_params (dict) :                   contains parameters for training and/or loading model
        train_mode (Boolean) :                  whether to train or load model
        inference_image (str) :                 address of image for inference (only if train_mode == True)
    """
    
    print("Saving params...")
    report_dir = "../train_unet/"+ str(datetime.now()) +"/"
    os.mkdir(report_dir)    # make directory
    model_params_file = open(report_dir + "model_params.txt", "w+")
    model_params_file.write(str(model_params))
    model_params_file.close()
    
    train_params_file = open(report_dir + "train_params.txt", "w+")
    train_params_file.write(str(train_params))
    train_params_file.close()
    
    
    print("Getting colors...")
    color_fname = train_params['colors']
    print(color_fname)
    colors = np.load(color_fname, allow_pickle=True, encoding='bytes')[0]
    num_colors = np.shape(colors)[0]
    print("We are using ", num_colors, " colors.")
    
    if train_mode is True or train_params['load_location'] == "":
        #   create model
        print("Building model...")
        model = UNet(model_params, num_colors).cuda()
        #   train model
        print("Training...")
        model = train(train_params, colors, model, report_dir=report_dir)
    else:
        print("Loading model...")
        checkpoint = torch.load(train_params['load_location'])
        model = checkpoint['model'].cuda()
        model.load_state_dict(checkpoint['state_dict'])
        
    #   freeze model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if inference_image != "":
        print("Evaluating model...")
        evaluate(model, inference_image, colors, save_dir=report_dir)
    print("Done.")

if __name__ == "__main__":
    
    
    train_params = {'download_dir':               '../data',
                    'image_classes':              [7],
                    'batch_size':                 100,
                    'num_epochs':                 200,
                    'learning_rate':              0.0001,
                    'learning_decay':             0.9,
                    'report_file':                'report',
                    'report_every_epoch':         10,
                    'report_every_batch':         100,
                    'colors':                     './colours/colour_kmeans16_cat7.npy',                         
                    'load_location':              '',       #   need to match num_colors
                    'categorical':                False             #   true if categorical colors, false if rgb                     
                    }
    
    model_params = {'kernel_size':                  3,
                    'num_filters':                  32,
                    'num_in_channels':              1,              #   1 if down-size to categorical colors, 3 if keep RGB
            }
    
    in_image = '/h/arnold/Desktop/NN/Colorization/image.png'
    main(model_params, train_params, train_mode=True, inference_image=in_image)
    
    train_params['colors'] = './colours/colour_kmeans32_cat7.npy'
    main(model_params, train_params, train_mode=True, inference_image=in_image)
    
    train_params['image_classes'] = [3]
    main(model_params, train_params, train_mode=True, inference_image=in_image)
    
    train_params['colors'] = './colours/colour_kmeans16_cat7.npy'
    main(model_params, train_params, train_mode=True, inference_image=in_image)
   