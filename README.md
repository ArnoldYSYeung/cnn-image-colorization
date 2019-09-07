# cnn-image-colorization
Using Convolutional Neural Networks to Colorize Greyscale Images

##  Summary
A convolutional neural network (CNN) architecture is designed to convert greyscale images to colorized RGB images.  The network is trained and evaluated on independent classes in the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  To simplify the task, training RGB images are saturated to pre-selected 16- and 32-color options.  Therefore, output colorized images are also restricted to these options. 

Once trained, new images may be inputted into the CNN for colorization.  Images similar to the training dataset (e.g., containing the same objects) work best with the CNN architecture.

##  Results
The CNN is trained with 2 classes in the CIFAR-10 dataset:horses and cats.  Experiments for each class were conducted with both the 16-color option and the 32-color option.

### Horses

### Cats

### Evaluation Image


##  Method


##  Execution
This project requires installation of the following packages:
- PyTorch
- Torchvision
- Numpy
- Matplotlib
- PIL
