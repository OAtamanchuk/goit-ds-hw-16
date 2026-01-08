## Description

This is a web application on Streamlit for classifying clothing images from the Fashion MNIST dataset using CNN and VGG16 models.
The application allows users to upload images, display classification results and class probabilities.
The application implements image upload (PNG/JPG/JPEG), their preprocessing, classification by the selected model, inference of the predicted class, confidence and probabilities, as well as graphs of loss and training accuracy of both models.
The models are trained on Fashion MNIST, Git LFS is used for large files.

Deploy: https://fashionmnistcnnvgg16.streamlit.app

## Technologies & Stack

- **Python** 
- **Streamlit** 
- **TensorFlow / Keras** 
- **Convolutional Neural Networks (CNN)**
- **VGG16 (Transfer Learning)** 
- **NumPy**
- **Pillow (PIL)** 
- **Matplotlib** 
- **Fashion MNIST dataset** 

## Functionality

The Fashion MNIST classifier web application provides the following functionality:
- Image upload in PNG, JPG, or JPEG format
- Model selection: choose between CNN or VGG16 pre-trained models for classification.
- Image preprocessing:
   - Grayscale conversion and inversion
   - Contrast enhancement
   - Centering and resizing for model input
   - RGB stacking for VGG16
- Classification output:
   - Predicted class
   - Confidence score (%)
   - Probabilities for all classes visualized as progress bars
- Preprocessed image visualization
- Training metrics visualization for the selected model.

## Links

- Deploy:

https://fashionmnistcnnvgg16.streamlit.app

- GitHub Repository:

https://github.com/OAtamanchuk/goit-ds-hw-16
