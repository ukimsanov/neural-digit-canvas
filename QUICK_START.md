# 🚀 Quick Start Guide

## What This Project Is
A complete ML system with multiple interfaces for MNIST digit recognition.

## 🎮 Option 1: Web Demo (Easiest)
```bash
cd /Users/kimsanov/Downloads/mnist-linear-classifier
source venv/bin/activate
python app.py
```
Then open: **http://localhost:7860**
- ✨ Draw digits with your mouse
- 🔮 See instant AI predictions
- 📊 Compare model confidence

## 🚀 Option 2: REST API (For Developers)
```bash
cd /Users/kimsanov/Downloads/mnist-linear-classifier
source venv/bin/activate
uvicorn api:app --reload
```
Then open: **http://localhost:8000/docs**
- 📡 Test API endpoints
- 📤 Upload images via HTTP
- 💾 Get JSON predictions

## 💻 Option 3: Command Line (For ML Engineers)
```bash
cd /Users/kimsanov/Downloads/mnist-linear-classifier
source venv/bin/activate

# Train models
python train.py --model linear --epochs 3

# Make predictions
python inference.py data/sample_digit.png --visualize

# Run tests
pytest tests/ -v
```

## 📓 Option 4: Jupyter Notebook (For Learning)
```bash
cd /Users/kimsanov/Downloads/mnist-linear-classifier
source venv/bin/activate
jupyter notebook mnist_linear_classifier.ipynb
```

## 🎯 What You'll Experience

### Web Demo
- 🎨 **Interactive drawing canvas**
- 🔮 **Real-time predictions** as you draw
- 📊 **Visual confidence scores**
- 🤖 **Model comparison** (Linear vs CNN)

### REST API
- 🌐 **Swagger UI** for testing
- 📤 **File upload** for predictions
- 📊 **JSON responses** with probabilities
- 🔧 **Health checks** and model info

### Command Line
- 📈 **Training progress bars**
- 📊 **Accuracy metrics** and loss curves
- 🎯 **Model evaluation** reports
- 📸 **Prediction visualizations**

## 🏆 This Shows You're A Pro
- ✅ Modern MLOps practices
- ✅ Production-ready deployment
- ✅ Multiple interface options
- ✅ Professional software engineering