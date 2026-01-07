## Description

This is a web application on Streamlit for classifying clothing images from the Fashion MNIST dataset using CNN and VGG16 models.
The application allows users to upload images, display classification results and class probabilities.
The application implements image upload (PNG/JPG/JPEG), their preprocessing, classification by the selected model, inference of the predicted class, confidence and probabilities, as well as graphs of loss and training accuracy of both models.
The models are trained on Fashion MNIST, Git LFS is used for large files.

Deploy: https://fashionmnistcnnvgg16.streamlit.app

## Technologies & Stack

- **Python** - main programming language  
- **Streamlit** - web application framework for interactive UI  
- **TensorFlow / Keras** - training and inference of neural network models  
- **Convolutional Neural Networks (CNN)** - custom model trained on Fashion MNIST  
- **VGG16 (Transfer Learning)** - fine-tuned model for image classification  
- **NumPy** - numerical computations and data processing  
- **Pillow (PIL)** - image loading and preprocessing  
- **Matplotlib** - visualization of training metrics (loss and accuracy)  
- **Fashion MNIST dataset** - image dataset for training and evaluation  

