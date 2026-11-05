# Cat vs Dog Image Classification Using CNN

## Project Overview

This project implements a custom Convolutional Neural Network (CNN) using PyTorch to classify images into two categories: Cat and Dog.

The model learns visual patterns and features from images during training and predicts whether a new uploaded image belongs to the Cat or Dog category.

The trained model is integrated into a Streamlit web application, allowing users to upload images and view predictions.

## Dataset

- Dataset: Kaggle Cats vs Dogs
- Total Images: 24,998
- Classes: Cat and Dog
- Image Dimensions: 128 × 128
- Training Data: 80%
- Testing Data: 20%

## CNN Architecture

The custom CNN consists of:

1. Input Layer – RGB images (128 × 128)
2. Convolutional Layer – 32 filters
3. ReLU Activation
4. Max Pooling
5. Convolutional Layer – 64 filters
6. ReLU Activation
7. Max Pooling
8. Flatten Layer
9. Fully Connected Layer – 128 neurons
10. Output Layer – 2 classes (Cat and Dog)

## Technologies Used

- Python
- PyTorch
- Torchvision
- CNN
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Pillow (PIL)

## Project Workflow

- Dataset loading and preprocessing
- Corrupted image checking
- Image resizing and tensor conversion
- Training and testing data splitting
- Custom CNN architecture development
- Model training using Adam optimizer
- Model evaluation using accuracy, precision, recall and F1-score
- Prediction on new images
- Model saving using PyTorch
- Streamlit web application development
- Deployment using Streamlit Community Cloud

## Web Application

The Streamlit application allows users to upload an image and receive a Cat or Dog classification prediction.

**Live Application:** [Cat vs Dog Classification](PASTE_YOUR_STREAMLIT_URL_HERE)

## Project Structure

```text
cat-dog-classification-CNN/
│
├── app.py
├── cat_dog_classification.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
└── saved_model/
    └── cat_dog_cnn.pth
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/murtuzaali87/cat-dog-classification-CNN.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## Author

**Mohammed Murtuza Ali**

AI & Data Science Student | Aspiring AI Engineer & Data Scientist

- **GitHub:** [github.com/murtuzaali87](https://github.com/murtuzaali87)
- **LinkedIn:** [linkedin.com/in/mohammedmurtuzaali](https://linkedin.com/in/mohammedmurtuzaali)