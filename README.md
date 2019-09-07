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

The colorization quality improves during training.  In the below validation images (epochs 0, 99, and 199), the greyscale images, colorized images, saturated images (ground truths), and original images are shown (top to bottom).

<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_horses/valid_e0.png" alt="Output of Epoch 0 for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_horses/valid_e0.png" alt="Output of Epoch 0 for 32 colors" width="400"/>
<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_horses/valid_e99.png" alt="Output of Epoch 99 for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_horses/valid_e99.png" alt="Output of Epoch 99 for 32 colors" width="400"/>
<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_horses/valid_e199.png" alt="Output of Epoch 199 for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_horses/valid_e199.png" alt="Output of Epoch 199 for 32 colors" width="400"/>

For more validation images obtained during the training process, see:
- https://github.com/ArnoldYSYeung/cnn-image-colorization/tree/master/train/16_horses
- https://github.com/ArnoldYSYeung/cnn-image-colorization/tree/master/train/32_horses

### Cats
Similarly for cats, we observe the following changes in loss for the 16-color and 32-color categories, respectively.

<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_cats/losses.svg" alt="Training loss for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_cats/losses.svg" alt="Training loss for 32 colors" width="400"/>

The colorization quality improves during training. However, it appears that colorization of cats is more difficult, given the greater diversity of fur colors than that of horses.  Instead, we observe that the most common cat color (i.e., brownish grey) is selected for most cats which do not have light (white) or dark (black) fur.

In the below validation images (epochs 0, 99, and 199), the greyscale images, colorized images, saturated images (ground truths), and original images are shown (top to bottom).

<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_cats/valid_e0.png" alt="Output of Epoch 0 for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_cats/valid_e0.png" alt="Output of Epoch 0 for 32 colors" width="400"/>
<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_cats/valid_e99.png" alt="Output of Epoch 99 for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_cats/valid_e99.png" alt="Output of Epoch 99 for 32 colors" width="400"/>
<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/16_cats/valid_e199.png" alt="Output of Epoch 199 for 16 colors" width="400"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_cats/valid_e199.png" alt="Output of Epoch 199 for 32 colors" width="400"/>

For more validation images obtained during the training process, see:
- https://github.com/ArnoldYSYeung/cnn-image-colorization/tree/master/train/16_cats
- https://github.com/ArnoldYSYeung/cnn-image-colorization/tree/master/train/32_cats

##  Discussion

A test image of a pair of horses is inputted into models trained for horses and cats independently.  From the images below, we see that the model trained for horses is able to select the correct color for the horse (i.e., brown), whereas the model trained for cats selected the most common cat color (i.e., brownish grey) for the horse. 

This suggests that, while both models can identify objects to-be-colored, training on similar images is important to capture the "most common" colors of the objects.  When an input is greyscale, information regarding the RGB scale is lost and model must compensate via its "intuition" of colors of similar objects.

<img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_horses/evaluate.png" alt="Test image for 32-color horses" width="600"/><img src="https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/train/32_cats/evaluate.png" alt="Test image for 32-color horses" width="600"/>


##  Method

The following CNN architecture is used:
- 2 Downsampling Convolutional Layers (2D Convolution, Batch Normalization, ReLU, Max Pooling)
- 1 Refactoring Convolutional Layer (2D Convolution, Batch Normalization, ReLU)
- 2 Upsampling Convolutional Layers (2D Convolution, Batch Normalization, ReLU, Upsampling)
- 1 Convolutional Layer (2D Convolution)

For training, the Adam optimizer and Cross Entropy Loss function were used.

While color regression within a color space is a viable option, I selected saturating the RGB images to a selected number of color categories, turning the task into a classification problem.  This (hopefully) ensures that the loss metric is a representation of the perception of color, instead of the distance within an arbitruary color space (e.g., RGB) which may not necessarily represent how humans perceive colors, psychologically (e.g., 1 color, not 3 combined) and biologically (e.g., cones do not map to color space). 

##  Execution
This project requires installation of the following packages:
- PyTorch 1.1.0
- Torchvision 0.3.0
- Numpy 1.16.4
- Matplotlib 3.1.0
- PIL 6.0.0

To run experiment, in [`src\color_classification.py`](https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/src/color_classification.py), set `train_params['image_classes']` to the CIFAR-10 classes to train the model on.  Indicate the location of the color numpy file to use in `train_params['colors']` and the model to load in `train_params['load_location']`.

When running function `main(...)`, set parameter `train_mode=True` for training and `train_mode=False` for inference.  For evaluating with a specific image, enter in the image location in the parameter `inference_image`.

### Problems?
One potential reason for low quality output images or errors may be due to the conversion of RGB, greyscale, and color categorical images.  When converted to a Numpy array, images may take values with the ranges 0 to 1, -1 to 1, or 0 to 255.  If the user encounters such problems, he/she should verify that the conversion scale is proper when calling function `normalize_array(...)` in [`src\utils.py`](https://github.com/ArnoldYSYeung/cnn-image-colorization/blob/master/src/utils.py).  (This will require some code debugging.)  I would make the code more robust, but no time :(
