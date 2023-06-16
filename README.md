# Bangkit-Machine-Learning

This repository contains a collection of resources used during the capstone project for Bangkit Machine Learning. The project focuses on building machine learning models for our application. For our machine learning needs, we have built image classifier models for plant disease detection and Soil detection 

<img src="https://github.com/Terrafarms/bangkit-machine-learning/assets/66078837/71a15c02-e1d8-49ed-8336-67b1de7e8a60"  width="400" height="700">
<img src="https://github.com/Terrafarms/bangkit-machine-learning/assets/66078837/adcae47f-6d06-4ba4-9b13-fea0facb5190"  width="400" height="700">

## Architecture

![WhatsApp Image 2023-06-16 at 10 30 48](https://github.com/Terrafarms/bangkit-machine-learning/assets/54931717/b76acca4-5333-4d48-96d3-c3b6c0f996c2)

*Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4510-4520).*

## Datasets
* [New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
* [Corn or Maize Leaf Disease Dataset](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)
* [Rice Leaf Disease Dataset](https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases)
* [Soil Type Dataset](https://www.kaggle.com/datasets/prasanshasatpathy/soil-types)


## Models

### Model Overview

The model architecture consists of the following layers:
- Convolutional layer with 32 filters and a ReLU activation function.
- Max pooling layer with a 2x2 pool size.
- Flatten layer to convert the 2D feature maps into a 1D feature vector.
- Dense layer with the number of units equal to the number of classes, using softmax activation.

### Data Processing

The dataset we used consists of the `Plant_Disease_Classifcation` model are images belonging to nine different classes of plant diseases, including Corn Blight, Corn Common Rust, Corn Gray Spot, Corn Healthy, Potato Blight, Potato Healthy, Rice Blight, Rice Brown Spot, and Rice Leaf Smut. While the dataset for the `Soil_Types_Classification` model are images belonging to five different classes of soil types, including Black Soil, Cinder Soil, Laterite Soil, Peat Soil, and Yellow Soil.

Data augmentation techniques are applied to increase the diversity and size of the dataset. The `ImageDataGenerator` class from TensorFlow is used for rescaling, rotation, zooming, flipping, shifting, shearing, and adjusting the brightness of the images.

### Model Training

The model is then trained using the augmented dataset. `Plant_Disease_Classifcation` model training is performed for 10 epochs with a batch size of 8. While `Soil_Types_Classification` model is performed for 25 epochs with a batch size of 4.  Two callbacks, EarlyStopping and ReduceLROnPlateau, are used for early stopping and learning rate reduction based on validation loss.

### Model Evaluation

The trained model is evaluated using the test dataset. The evaluation provides the loss and accuracy scores of the model on the test dataset. Additionally, a sample of images from the test dataset is used to demonstrate the model's predictions. The predicted class and confidence score are displayed for each image.

### Model Saving and Conversion

The trained model is saved in the HDF5 format as `model.h5` for future use. To integrate the model with Android applications, the model is converted to the TensorFlow Lite (TFLite) format using the TFLite Converter. The TFLite model is saved as `model.tflite` for deployment on resource-constrained devices.

In addition to the TFLite model, a metadata file (`metadata.txt`) is provided. The metadata file contains information about the model, such as input and output tensor names, input and output types, and model descriptions. It serves as a reference for integrating the TFLite model into applications.

## Requirements

To run the notebook and utilize the model, the following dependencies are required:
- Tensorflow
- Keras
- Matplotlib
- NumPy
- PIL
- psutil

Make sure to install these dependencies before running the notebook.

## Usage

1. Clone the repository

```bash
git clone github.com/Terrafarms/bangkit-machine-learning.git
```

2. Install the required dependencies in your Google Colab/Jupyter Notebook

```bash
pip install tensorflow keras matplotlib numpy pillow psutil
```

3. Navigate to the repository `Notebooks` directory and open the notebooks

4. Run the cells in the notebook to train the model and evaluate its performance.
   
5. Save the trained model as model.h5 for future use.
   
6. Convert the model to TFLite format using the provided code and save it as model.tflite.

7. Use the saved model for inference in your applications. Refer to the metadata file (metadata.txt) for integration details.

## Contributing

Contributions to this project are welcome. If you have any ideas, suggestions, or improvements, please submit a pull request. Make sure to follow the existing coding style and guidelines.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and distribute the code as per the license terms.

## Contact

For any inquiries or feedback, please contact the project team at contact@terrafarms.id

