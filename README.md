# Bangkit-machine-learning

For our machine learning we build image classifier for plant desease detection and Soil detection

<img src="https://github.com/Terrafarms/bangkit-machine-learning/assets/66078837/71a15c02-e1d8-49ed-8336-67b1de7e8a60"  width="400" height="700">
<img src="https://github.com/Terrafarms/bangkit-machine-learning/assets/66078837/adcae47f-6d06-4ba4-9b13-fea0facb5190"  width="400" height="700">

## Architecture

![image](https://github.com/Terrafarms/bangkit-machine-learning/assets/66078837/986cd74b-3062-43bc-84de-b5a125663e97)


## Datasets
* [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
* [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
* [Rice Leaf Disease](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)



## Models

### Plant Disease Classification Model

The model architecture consists of the following layers:
    - Convolutional layer with 32 filters and a ReLU activation function.
    - Max pooling layer with a 2x2 pool size.
    - Flatten layer to convert the 2D feature maps into a 1D feature vector.
    - Dense layer with the number of units equal to the number of classes, using softmax activation.

The dataset we used consists of images belonging to nine different classes of plant diseases, including Corn Blight, Corn Common Rust, Corn Gray Spot, Corn Healthy, Potato Blight, Potato Healthy, Rice Blight, Rice Brown Spot, and Rice Leaf Smut.

Data augmentation techniques are applied to increase the diversity and size of the dataset. The ImageDataGenerator class from TensorFlow is used for rescaling, rotation, zooming, flipping, shifting, shearing, and adjusting brightness of the images.

The model is then trained using the augmented dataset. The training is performed for 10 epochs with a batch size of 8. Two callbacks, EarlyStopping and ReduceLROnPlateau, are used for early stopping and learning rate reduction based on validation loss.

The trained model is saved in the HDF5 format as model.h5 for future use. To integrate the model with android applications, the model is converted to the TensorFlow Lite (TFLite) format using the TFLite Converter. The TFLite model is saved as model.tflite for deployment on resource-constrained devices.