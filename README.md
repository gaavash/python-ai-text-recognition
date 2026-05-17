# 🧠 Handwritten Digit Recognition using Neural Networks

A deep learning project built with Python, PyTorch, and OpenCV for handwritten digit recognition using the MNIST dataset and custom JPG digit images.

This project demonstrates the implementation of a Feedforward Neural Network (FNN) for image classification, including custom image preprocessing, dataset handling, neural network training, evaluation, and prediction generation.

---

# 🚀 Features

- ✅ Handwritten digit recognition (0–9)
- 🖼️ Custom JPG image loading and preprocessing
- 🧠 Feedforward Neural Network implementation
- 📚 MNIST dataset training
- 🔄 Data augmentation using image rotation
- 📊 Accuracy and loss evaluation
- ⚡ PyTorch-based model training
- 🧾 Automatic result export to text file

---

# 🛠️ Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- TorchVision
- Neural Networks
- Machine Learning
- Computer Vision

---

# 📂 Project Structure

```bash
project-folder/
│
├── images/
│   ├── digit_0.jpg
│   ├── digit_1.jpg
│   ├── ...
│   ├── digit_9.jpg
│
├── data/                     # MNIST dataset
├── assignment_results.txt    # Generated results
├── main.py                   # Main source code
```

---

# 🧠 Neural Network Architecture

The project uses a simple Feedforward Neural Network:

```bash
Input Layer:     784 neurons (28x28 image)
Hidden Layer 1:  128 neurons
Hidden Layer 2:   64 neurons
Output Layer:     10 neurons (digits 0-9)
```

### Activation Function
- Sigmoid

### Loss Function
- CrossEntropyLoss

### Optimizer
- SGD (Stochastic Gradient Descent)

---

# 📖 How It Works

## 1️⃣ Image Preprocessing
- Loads custom JPG digit images
- Converts images to grayscale
- Resizes images to 28×28 pixels
- Inverts colors when necessary
- Normalizes pixel values

## 2️⃣ Model Training
- Downloads and trains on the MNIST dataset
- Uses mini-batch gradient descent
- Performs multiple training epochs

## 3️⃣ Fallback Training
If MNIST download fails:
- Uses custom image augmentation
- Rotates images randomly
- Generates synthetic training data

## 4️⃣ Prediction & Evaluation
- Predicts handwritten digits
- Calculates loss and accuracy
- Displays prediction results
- Saves outputs to a text file

---

# ▶️ Installation & Setup

## Clone Repository

```bash
git clone <repository-url>
cd project-folder
```

## Install Dependencies

```bash
pip install torch torchvision opencv-python numpy
```

## Run the Program

```bash
python main.py
```

---

# 📊 Output Example

```bash
Final Test Accuracy: 97.50%
Custom Images Accuracy: 9/10 = 90.0%
```

The program also generates:

```bash
assignment_results.txt
```

containing:
- Final loss
- Predictions
- Accuracy results

---

# 🎯 Learning Outcomes

Through this project, I learned:

- Neural network fundamentals
- Image preprocessing techniques
- Handwritten digit classification
- PyTorch model development
- Dataset handling with MNIST
- Model training and evaluation
- Data augmentation methods
- Computer vision basics

---

# 📌 Future Improvements

- Convolutional Neural Network (CNN) implementation
- Real-time webcam digit recognition
- GUI interface for digit drawing
- Model saving and loading
- GPU acceleration support
- Higher accuracy optimization
