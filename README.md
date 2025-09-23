# MNIST Linear Classifier

A simple yet effective implementation of linear classification for handwritten digit recognition using PyTorch. This project demonstrates that even basic neural network architectures can achieve solid performance on classic computer vision tasks.

## 🎯 Project Overview

This project implements a single-layer linear classifier to classify handwritten digits (0-9) from the MNIST dataset. The focus is on understanding neural network fundamentals without the complexity of deep architectures, proving that simplicity can be powerful.

### Key Features

- **Minimalist Architecture**: Single linear layer (784 → 10) for direct classification
- **Clean Implementation**: Well-documented PyTorch code with clear data flow
- **Comprehensive Visualization**: Training progress tracking and prediction analysis
- **Solid Performance**: Achieves ~92-93% accuracy on the test set

## 🚀 Results

The linear classifier achieves approximately **92-93% accuracy** on the MNIST test set, demonstrating that linear models can be surprisingly effective for certain classification tasks when the feature space allows for linear separability.

## 🏗️ Architecture

```
Input: 28×28 grayscale images → Flatten → Linear Layer → 10 classes
       (784 features)                    (784→10)      (digits 0-9)
```

**Model Details:**
- **Parameters**: 7,850 total (784×10 weights + 10 biases)
- **Activation**: None (raw logits for CrossEntropyLoss)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: SGD with learning rate 0.01

## 📊 Dataset

- **MNIST Handwritten Digits**
  - Training: 60,000 samples
  - Testing: 10,000 samples
  - Image size: 28×28 pixels (grayscale)
  - Classes: 10 (digits 0-9)

## 🛠️ Implementation

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy
```

### Usage

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mnist-linear-classifier
   ```

2. **Run the notebook**
   ```bash
   jupyter notebook mnist_linear_classifier.ipynb
   ```

The notebook will automatically:
- Download the MNIST dataset
- Preprocess the data (normalization, batching)
- Train the linear classifier
- Evaluate performance and visualize results

### Project Structure

```
mnist-linear-classifier/
├── mnist_linear_classifier.ipynb  # Main implementation notebook
├── data/                          # MNIST dataset (auto-downloaded)
│   └── MNIST/
└── README.md                      # This file
```

## 📈 Training Process

The model trains for 5 epochs using:
- **Batch size**: 64
- **Learning rate**: 0.01
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Loss function**: CrossEntropyLoss

Training progress is visualized with loss and accuracy plots.

## 🔍 Key Insights

1. **Linear Models Work**: Despite simplicity, linear classification achieves good results on MNIST
2. **Proper Initialization**: Small random weights (σ=0.01) ensure stable training
3. **Data Preprocessing**: Normalization to [-1,1] improves convergence
4. **Batch Processing**: DataLoader with batching improves training efficiency

## 🚀 Potential Improvements

- **Regularization**: Add L1/L2 regularization to prevent overfitting
- **Learning Rate Scheduling**: Implement adaptive learning rates
- **Different Optimizers**: Experiment with Adam or RMSprop
- **Data Augmentation**: Add rotation/translation for better generalization
- **Deep Architecture**: Add hidden layers for increased capacity

## 📝 Technical Details

The implementation showcases:
- PyTorch tensor operations and automatic differentiation
- Proper train/test data splitting and evaluation
- Gradient descent optimization
- Loss function implementation for classification
- Model evaluation and prediction visualization

## 🤝 Contributing

Feel free to fork this repository and experiment with different approaches! Some ideas:
- Try different optimizers and learning rates
- Implement regularization techniques
- Add more sophisticated visualization
- Compare with other simple baselines

## 📄 License

This project is open source and available under the MIT License.

---

*This project demonstrates that understanding fundamentals is just as important as working with complex architectures. Sometimes, simple solutions are the most elegant.*