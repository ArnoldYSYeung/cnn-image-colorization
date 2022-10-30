"""
Utility functions.
Author :                    Arnold Yeung
Date :                      September 6th, 2019
"""
import os

import torch
import numpy as np
import urllib.request
import tarfile

import matplotlib.pyplot as plt

def rgb_to_greyscale(image, max_pixel = 1.0):
    """
    Converts a RGB image tensor to greyscale.
    Arguments:
        image (torch.tensor) :             tensor of RGB images (num_samples * 3 * width * height)
        max_pixel (float) :             maximum possible value of pixel 
    Returns:
        greyscale version of image (num_samples * 1 * width * height)
    """
    image = image/max_pixel
    grey = torch.mean(image, dim=1).unsqueeze(dim=1) 
    
    return grey

def normalize_array(img, minmax=[0, 255], scale_range=[-1, 1], dtype=float):
    """
    Scales the values of an image array to between -1 to 1.
    Arguments:
        img (np.array) :                    image array to scale
        minmax (list[int, int]) :           list containing the minimum and maximum values of img
        scale_range (list[int, int]) :      list containing the range to scale to
        dtype(datatype) :                   data type of values of returned image array
    Returns:
        img (np.array) :                    array with normalized values
    """
    
    if len(minmax) != 2 or len(scale_range) != 2:
        raise ValueError("Input lists are not of length 2.")
    
    min_value, max_value = minmax
    min_scale, max_scale = scale_range
    normalizer = (max_value-min_value)/(max_scale-min_scale)
    img = (img-min_value)/normalizer + min_scale
    img = img.astype(dtype)             #   change the data type of the array
    return img

def download_file(fname, url, save_dir, untar=False, extract=False):
    """
    Download file from a url.
    Arguments:
        fname (str) :                   filename
        url (str) :                     url to download file from
        save_dir (str) :                directory to save file to
        untar (Boolean) :               add 'tar.gz' at the end of file
        extract (Boolean) :             whether to extract file or not
    """
 
    if save_dir[-1] != "/":
        save_dir + "/"
    
    fpath = os.path.join(save_dir, fname)
    if untar:
        fpath += '.tar.gz'
    
    urllib.request.urlretrieve(url, fpath)
    
    #   Extract file
    if extract:
        with tarfile.open(fpath) as archive:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(archive, save_dir)
            
def map_rgb_to_categories(images, colors):
    """
    Maps the values of an image array to categorical colors.
    Arguments:
        images (torch.tensor) :             images array (num_samples * 3 * width * height)
        colors (np.array) :                 color array of different color categories and their RGB values
    Returns:
        images (np.array) :            image array mapped to categorical color values 
                                            (num_samples * 1 * width * height)
    """
    #   normalizes image values to 0 and 1
    images = torch.tensor(normalize_array(images.cpu().numpy(), minmax=[-1, 1], scale_range=[0, 1]))

    if images.shape[0] < 100:
        return _map_rgb_to_categories(images, colors)
    
    num_samples = images.shape[0]
    batch_size = 100

    list_images = []
    
    for i in range(0, num_samples, batch_size):
        cat_images = _map_rgb_to_categories(images[i:i+batch_size, :, :, :], colors)
        list_images.append(cat_images)        
    return np.concatenate(list_images, axis=0)
        
def _map_rgb_to_categories(images, colors):
    """
    See map_rgb_to_categories().  This function is memory intensive, so input data is broken down to
    smaller batches in map_rgb_to_categories().
    """
    num_colors = colors.shape[0]
    colors = np.reshape(colors, (num_colors, 1, 3, 1, 1))       #   RGB

    images = np.expand_dims(images.cpu(), axis=0)                  #   (1, num_samples * 3 * width * height)
    distances = np.linalg.norm(images-colors, axis=2)       #   matrix norm along the RGB axis
    
    images = np.expand_dims(np.argmin(distances, axis=0), axis=1)
    return images

def map_categories_to_rgb(images, colors):
    """
    Maps the values of a categorical color array to an RGB array.
    Arguments:
        images (torch.tensor) :             images array (num_samples * 1 * height * width)
        colors (np.array) :                 color array of different color categories and their RGB values
    Returns:
        images (torch.tensor) :             image array mapped to RGB color values (num_samples * 3 * height * width)
    """
    images = images.cpu().numpy()
    image_list = []
    for sample in range(images.shape[0]):
        image = colors[images[sample, :, :, :]]                                 #   1 * height * width * 3 (rgb)
        image_list.append(image)
    images = np.array(image_list)
    images = images.transpose((0, 4, 2, 3, 1))                 #   num_samples * 3 * height * width * 1
    images = images[:, :, :, :, 0]                             #   num_samples * 3 * height * width
    return torch.tensor(images)

def plot_time_series(data, yrange=[], colors=[], markers=[], 
                     linestyles=[], legend=[], title='', save_loc='', 
                     save_name='', save_format='svg'):
    """
    Produces a plot of all time series data listed in data.  Each set of time series
    data is stored as a list, within input data.
    At most 7 data sets per plot.
    Arguments:
        data (list[list[], ]) :             time series data sets (stored as lists)
                                            to be plotted
        yrange (list[float, float]) :       lower and upper bounds of y-axis
        colors (list[str, ]) :              colors in order of data sets
        markers (list[str,]) :              marker styles in order of data sets
        linestyles (list[str, ]) :          linestyles in order of data sets
        legend (list[str, ]) :              names in order of data sets
        title (str) :                       title of plot
        save_loc (str) :                    directory to save figure in
        save_name (str) :                   file name
        save_format (str) :                 file extension of output file ['png', 'pdf', 'svg']
    Returns:
        None
    """
    #   default styles
    default_colors = ['b', 'r', 'k', 'c', 'm', 'g', 'y']
    default_markers = ['o', '.', 's', 'P', 'D', '^', '*']
    default_linestyles = ['solid'] * 7
    has_legend = True
    
    if not colors:
        colors = default_colors[:len(data)]
    if not markers:
        markers = default_markers[:len(data)]
    if not linestyles:
        linestyles = default_linestyles[:len(data)]
    if not legend:
        has_legend = False

    #   make sure that all lengths are the same
    if len(colors) == len(markers) == len(linestyles) == len(data):
        pass
    else:
        raise IndexError("Lengths of inputs not the same.")
    
    fig = plt.figure(1)
    plt.title(title)
    for i, dataset in enumerate(data):
        if has_legend:
            plt.plot(range(0, len(dataset)), dataset, color=colors[i], marker=markers[i],
                     linestyle=linestyles[i], label=legend[i])
        else:
            plt.plot(range(0, len(dataset)), dataset, color=colors[i], marker=markers[i],
                     linestyle=linestyles[i])
    fig.legend()
    
    if save_loc != '' and save_name != '' and save_format != '':
        #   save the figure
        plt.savefig(save_loc+save_name+'.'+save_format)
    
    plt.close(fig)

def load_cifar10(data_dir, batch_size, labels=None):
    """
    Loads CIFAR10 data into training and validation set.
    Arguments:
        data_dir (str) :                directory to save data into
        batch_size (int) :              size of each batch
        labels (list[int, ]) :          labels in CIFAR10 to include (if None, include all labels)
    Returns:
        A training dataloader and a validation dataloader.
    """
    
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    training_set = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    valid_set = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    if labels is None:
        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    else:
        #   get indices of matching labels
        training_indices = []
        valid_indices = []
        for i, sample in enumerate(training_set):
            if sample[1] in labels:
                training_indices.append(i)
        for j, sample in enumerate(valid_set):
            if sample[1] in labels:
                valid_indices.append(j)
        
        train_loader = DataLoader(training_set, batch_size=batch_size,
                                  num_workers=2, sampler=SubsetRandomSampler(training_indices))
        valid_loader = DataLoader(valid_set, batch_size=batch_size,
                                  num_workers=2, sampler=SubsetRandomSampler(valid_indices))
    
    return train_loader, valid_loader

def plot_image_set(images, predictions, ground_truths, real_images, path, greyscale=True):
    """
    Plot a set of 3 image tensors.
    Arguments: 
        images (torch.tensor) :              greyscale input images (num_samples * (1 or 3) * height * width)
        predictions (torch.tensor) :         output images with categorized rgb colors (num_samples * 3 * height * width)
        ground_truths (torch.tensor) :       ground truth images with categorized rgb colors (num_samples * 3 * height * width)
        real_images (torch.tensor) :         ground truth images with uncategorized rgb colors (num_samples * 3 * height * width)
        path (str) :                         directory and filename to save figure
        greyscale (Boolean) :                whether images are greyscale (image.shape[1]=1) or rgb (image.shape[1]=3)  
    """
    
    """
    if images.shape[0] != predictions.shape[0] or predictions.shape[0] != ground_truths.shape[0]:
        raise ValueError("The number of samples are not the same.")
    """
    num_samples = images.shape[0]
    height = images.shape[2]
    width = images.shape[3]
    
    #   reshape to num_samples * height * width * 3
    ground_truths = ground_truths.permute((0, 2, 3, 1))
    predictions = predictions.permute((0, 2, 3, 1))
    real_images = real_images.permute((0, 2, 3, 1))
    
    #   convert greyscale to RGB
    #images = torch.stack((images, images, images), dim=-1).squeeze()      #   replicate greyscale image 3 times (num_samples * height * width * 3)
    images = torch.stack((images, images, images), dim=-1) 
    max_value = torch.max(images).item()
    min_value = torch.min(images).item()
    images = torch.tensor(normalize_array(images.cpu().numpy(), minmax=[min_value, max_value], 
                                          scale_range=[0, 255], dtype=int))
    
    #   normalize to 0 to 255  
    
    predictions = torch.tensor(normalize_array(predictions.cpu().numpy(), minmax=[0, 1], 
                                               scale_range=[0, 255], dtype=int)).int()
    ground_truths = torch.tensor(normalize_array(ground_truths.cpu().numpy(), minmax=[0, 1], 
                                               scale_range=[0, 255], dtype=int)).int()
    real_images = torch.tensor(normalize_array(real_images.cpu().numpy(), minmax=[-1, 1], 
                                               scale_range=[0, 255], dtype=int)).int()
    
    #   first sample to plot  - concatenate along height
    concat_images = images[0, :, :, :].squeeze().int()            #   height * width * 3
    concat_images = torch.cat((concat_images, predictions[0, :, :, :]), dim=0)            #  (2*height) * width * 3
    concat_images = torch.cat((concat_images, ground_truths[0, :, :, :]), dim=0)          #   (3*height) * width * 3
    concat_images = torch.cat((concat_images, real_images[0, :, :, :]), dim=0)          #   (4*height) * width * 3
        
    for sample in range(1, images.shape[0]):     #   for each sample
        image = images[sample, :, :, :].squeeze().int()
        image = torch.cat((image, predictions[sample, :, :, :]), dim=0)
        image = torch.cat((image, ground_truths[sample, :, :, :]), dim=0)
        image = torch.cat((image, real_images[sample, :, :, :]), dim=0)
        concat_images = torch.cat((concat_images, image), dim=1)    #   concatenate horizontally
    
    #   reshape to (3*height) * (2*width) * 3
    concat_images = concat_images.numpy()
    
    #   concat_images in shape 3 * (3*height) * (2*width)
    plt.imshow(concat_images)
    plt.savefig(path)               #   must go before plt.show() to keep image
    plt.show()
    
