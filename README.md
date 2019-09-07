# cnn-image-colorization
Using Convolutional Neural Networks to Colorize Greyscale Images

##  Description
A convolutional neural network (CNN) architecture is designed to convert greyscale images to colorized RGB images.  The network is trained and evaluated on independent classes in the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Training RGB images are saturated to pre-selected 16- and 32-color options.  Therefore, output colorized images are also restricted to these options. 

Once trained, new images may be inputted into the CNN for colorization.  Images similar to the training dataset (e.g., containing the same objects) work best with the CNN architecture.

##  Results
The CNN is trained with 2 classes in the CIFAR-10 dataset: horses and cats.  Experiments for each class were conducted with both the 16-color option and the 32-color option.

### Horses
After training for 200 epochs, I observed the following changes in loss for the 16-color and 32-color categories, respectively:
<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_horses/losses.svg" alt="Training loss for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_horses/losses.svg" alt="Training loss for 32 colors" width="400"/>

The colorization quality improves during training.  In the below images, the greyscale images, colorized images, saturated images (ground truths), and original images are shown (top to bottom).

<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_horses/valid_e0.png" alt="Output of Epoch 0 for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_horses/valid_e0.png" alt="Output of Epoch 0 for 32 colors" width="400"/>

<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_horses/valid_e99.png" alt="Output of Epoch 99 for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_horses/valid_e99.png" alt="Output of Epoch 99 for 32 colors" width="400"/>

<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_horses/valid_e199.png" alt="Output of Epoch 199 for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_horses/valid_e199.png" alt="Output of Epoch 199 for 32 colors" width="400"/>


### Cats

### Evaluation Image


##  Method

While color regression within a color space is a viable option, I selected saturating the RGB images to a selected number of color categories, turning the task into a classification problem.  This (hopefully) ensures that the loss metric is a representation of the perception of color, instead of the distance within an arbitruary color space (e.g., RGB) which may not necessarily represent how humans perceive colors, psychologically (e.g., 1 color, not 3 combined) and biologically (e.g., cones do not map to color space). 

##  Execution
This project requires installation of the following packages:
- PyTorch
- Torchvision
- Numpy
- Matplotlib
- PIL
