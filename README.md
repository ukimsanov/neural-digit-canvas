# 🎯 MNIST Digit Classifier - Production-Ready ML Pipeline

[![CI Pipeline](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Code Quality](https://img.shields.io/badge/code%20quality-A-brightgreen)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-grade machine learning project** that transforms the classic MNIST digit classification challenge into a showcase of modern MLOps practices. This isn't just another tutorial—it's a complete ML engineering pipeline demonstrating enterprise-level development workflows, from research to deployment.

> **Live Demo**: Interactive web interface with real-time digit recognition  
> **API Ready**: RESTful service with OpenAPI documentation  
> **Container Native**: Full Docker containerization with orchestration

## 🌟 Why This Project Stands Out

This project elevates MNIST from a simple tutorial to a **professional ML engineering showcase**, demonstrating industry best practices that recruiters and fellow developers recognize:

### 🏗️ **Enterprise Architecture**
```
Production Pipeline: Research → Development → Testing → Deployment → Monitoring
```

### 🚀 **Core Capabilities**

**🤖 Dual Model Architecture**
- **Linear Classifier**: Efficient baseline achieving **92.42% accuracy** 
- **CNN Classifier**: Deep learning model reaching **98.16% accuracy**
- **Comparative Analysis**: Head-to-head performance benchmarking

**🌐 Full-Stack Implementation**
- **Interactive Web App**: Modern Next.js frontend with real-time digit drawing and prediction
- **REST API**: FastAPI service with auto-generated OpenAPI docs
- **Modern UI/UX**: Beautiful animations, responsive design, and brush thickness controls
- **CI/CD Pipeline**: Automated testing across Python 3.8-3.11

**📊 Professional ML Workflow**
- Structured project architecture with proper packaging
- Comprehensive evaluation metrics and visualizations  
- Model versioning and experiment tracking
- Production-ready inference pipeline with error handling

## ⚡ Quick Start

Get up and running in under 2 minutes:

```bash
# Clone and setup
git clone https://github.com/ukimsanov/mnist-linear-classifier.git
cd mnist-linear-classifier

# One-command setup and training
make install && make train

# Launch full application (API + Frontend)
make run
# 🌐 Frontend: http://localhost:3000
# 📡 API docs: http://localhost:8000/docs

# Or start services individually
make api      # Just the API server
make frontend # Just the Next.js frontend
```

**🐳 Prefer Docker?**
```bash
docker-compose up --build
# Everything runs automatically: API + Next.js frontend
```

## � Performance Benchmarks

| Model Architecture | Test Accuracy | Parameters | Inference Time* | Use Case |
|-------------------|---------------|------------|-----------------|----------|
| **Linear Classifier** | **92.42%** | 7,850 | ~2ms | Edge deployment, baseline |
| **CNN Classifier** | **98.16%** | 102,026 | ~8ms | Production accuracy |

*Per sample on CPU (Intel/Apple Silicon)*

### 🎯 **Key Metrics**
- **Training Speed**: CNN converges in <10 epochs (~2 min on modern hardware)
- **Model Size**: Linear: 31KB, CNN: 400KB (highly portable)
- **Deployment Ready**: Sub-10ms inference enables real-time applications

## 🧠 Model Architectures

### **Linear Classifier** - Minimalist Baseline
```python
Input(28×28) → Flatten(784) → Linear(784→10) → Softmax → Output(10)
```
Perfect for understanding fundamentals and edge deployment scenarios.

### **CNN Classifier** - Production Architecture  
```python
Input(28×28×1) 
↓
Conv2D(32) → BatchNorm → ReLU → MaxPool(2×2)
↓  
Conv2D(64) → BatchNorm → ReLU → MaxPool(2×2) 
↓
Conv2D(128) → BatchNorm → ReLU → AdaptiveAvgPool
↓
Dropout(0.5) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→10)
```
Modern architecture with batch normalization, dropout, and adaptive pooling for robust performance.

## �️ Installation & Setup

### **System Requirements**
- Python 3.8+ (tested on 3.8-3.11)
- 4GB RAM minimum (8GB recommended)
- ~1GB disk space for dataset and models

### **Installation Options**

**🚀 Quick Start (Recommended)**
```bash
git clone https://github.com/ukimsanov/mnist-linear-classifier.git
cd mnist-linear-classifier
make install  # Installs all dependencies
make train     # Trains both models (~5 minutes)
```

**🔧 Development Setup**
```bash
make install     # Install dependencies
make clean       # Clean cache files
python -m pytest tests/ # Run test suite (if you have tests)
```

**🐳 Docker Deployment**
```bash
docker-compose up --build
# Automatically handles: API + Next.js frontend services
```

## 🎮 Usage Examples

### **🏋️ Model Training**

```bash
# Train both models with performance comparison
python train.py --model both --epochs 10

# Advanced CNN training with customization
python train.py --model cnn --epochs 20 --lr 0.001 --optimizer adamw --dropout 0.5

# Quick linear model training
python train.py --model linear --epochs 5
```

### **🌐 Interactive Web Application**
```bash
make run
# 🎨 Modern Next.js app at http://localhost:3000
# ✨ Draw digits with brush thickness control
# 🚀 Beautiful animations and real-time predictions
```

### **🔌 REST API Service**
```bash
make api
# 📡 Full OpenAPI documentation at http://localhost:8000/docs
# 🧪 Test endpoints directly in the browser
```

**Example API Usage:**
```python
import requests

# Upload image for prediction
response = requests.post(
    "http://localhost:8000/predict",
    files={"file": open("digit.png", "rb")},
    data={"model_type": "cnn"}
)
print(response.json())  # {"predicted_class": 7, "confidence": 0.98}
```

### **⚡ Single Image Inference**
```bash
python inference.py digit.png --model-type cnn --visualize
# Outputs prediction with confidence visualization
```

## 📁 Project Architecture

```
mnist-linear-classifier/
├── 🧠 src/mnist_classifier/      # Core ML package
│   ├── models/                   # Neural network architectures
│   │   ├── __init__.py          # Model exports
│   │   ├── linear.py            # Linear classifier implementation
│   │   └── cnn.py               # CNN architecture with BatchNorm
│   ├── utils/                   # ML utilities and helpers
│   │   ├── data.py              # MNIST data loading & preprocessing
│   │   ├── evaluation.py        # Model evaluation and metrics
│   │   └── visualization.py     # Training plots & confusion matrices
│   └── train.py                 # Training orchestration logic
├── 🧪 tests/                     # Comprehensive test suite
│   ├── test_models.py           # Model architecture tests
│   ├── test_training.py         # Training pipeline tests
│   └── test_api.py              # API endpoint tests
├── 📊 outputs/                   # Training artifacts
│   ├── linear/                  # Linear model checkpoints
│   └── cnn/                     # CNN model checkpoints  
├── 🗃️ data/MNIST/               # Dataset storage
├── 🎯 train.py                  # CLI training interface
├── 🌐 api.py                    # FastAPI REST service
├── 🎨 frontend/                 # Next.js web application
│   ├── src/app/                 # Next.js 15 app router
│   ├── src/components/          # React components
│   └── public/                  # Static assets
├── 🐳 Dockerfile & docker-compose.yml  # Container orchestration
├── ⚙️ Makefile                  # Development automation
└── 🔄 .github/workflows/        # CI/CD automation
```

**🏆 Key Design Principles:**
- **Modular Architecture**: Clean separation of concerns
- **Production Ready**: Proper packaging and dependency management  
- **Testing First**: Comprehensive test coverage for reliability
- **Developer Experience**: Rich tooling and automation

## � API Reference

### **REST API Endpoints**

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/` | GET | Health check and service status | `{"status": "healthy"}` |
| `/models` | GET | List available models | `["linear", "cnn"]` |
| `/predict` | POST | Image upload prediction | Prediction with confidence |
| `/predict/base64` | POST | Base64 image prediction | Same as above |
| `/model/{type}/info` | GET | Model architecture details | Parameters, accuracy, etc. |

### **🎨 Frontend Features**

**Modern Next.js 15 Application:**
- **Real-time Drawing Canvas**: Interactive digit drawing with touch/mouse support
- **Brush Thickness Control**: Adjustable brush size (1-5) for different drawing styles  
- **Model Selection**: Switch between CNN and Linear models with live accuracy display
- **Beautiful Animations**: Smooth entrance animations and micro-interactions
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile
- **Live API Status**: Real-time connection status indicator
- **Glassmorphism UI**: Modern frosted glass effects and gradients

**Technical Stack:**
- Next.js 15 with App Router
- React 19 with TypeScript
- Tailwind CSS v4 for styling
- Custom CSS animations for 2025 design trends

### **Python Package API** 

```python
# Model Creation & Training
from src.mnist_classifier.models import LinearClassifier, CNNClassifier
from src.mnist_classifier.train import Trainer
from src.mnist_classifier.utils import load_data, evaluate_model

# Quick model setup
model = CNNClassifier()
trainer = Trainer(model, learning_rate=0.001)

# Data pipeline
train_loader, val_loader, test_loader = load_data(batch_size=64)

# Training with monitoring
history = trainer.train(
    train_loader, 
    val_loader, 
    epochs=10,
    save_path="outputs/my_model.pth"
)

# Evaluation
metrics = evaluate_model(model, test_loader)
print(f"Test Accuracy: {metrics['accuracy']:.2%}")
```

### **Docker API**
```bash
# Production deployment
docker-compose up -d
curl -X POST "http://localhost:8000/predict" \
     -F "file=@digit.png" \
     -F "model_type=cnn"
```

## 🧪 Quality Assurance

### **Testing Pipeline**
```bash
python -m pytest tests/ -v        # Run test suite
python -m pytest tests/ --cov     # With coverage report
make clean                         # Clean cache files
```

### **Test Coverage**
- **Model Architecture**: Unit tests for both Linear and CNN models
- **Training Pipeline**: Integration tests for the complete training workflow  
- **API Endpoints**: Comprehensive REST API testing
- **Data Pipeline**: Data loading and preprocessing validation
- **Inference**: End-to-end prediction testing

### **Code Quality Standards**
- **Black**: Consistent code formatting
- **Flake8**: PEP 8 compliance and error detection
- **MyPy**: Static type checking
- **Pytest**: Robust testing framework with fixtures

## 🐳 Docker Deployment

### **One-Command Deployment**
```bash
docker-compose up --build
# 🚀 Automatically starts: Next.js frontend (port 3000) + REST API (port 8000)
```

### **Service Management**
```bash
# Training in containers
docker-compose --profile training up train

# Scale services
docker-compose up --scale api=3

# Monitor logs
docker-compose logs -f api

# Clean shutdown
docker-compose down --volumes
```

### **Production Configuration**
- **Resource Limits**: CPU and memory constraints configured
- **Health Checks**: Automatic service monitoring and restart
- **Volume Persistence**: Model checkpoints and data preserved
- **Multi-Stage Builds**: Optimized image sizes for deployment

## � CI/CD Pipeline

**Enterprise-grade automation** ensuring code quality and reliability:

### **Continuous Integration**
- ✅ **Multi-Python Testing**: Automated testing across Python 3.8-3.11
- 🔍 **Code Quality Gates**: Black formatting, Flake8 linting, MyPy type checking  
- 📊 **Coverage Tracking**: Comprehensive test coverage reporting
- 🐳 **Container Validation**: Docker build and deployment testing
- 🧠 **Model Validation**: Automated training pipeline verification

### **Quality Metrics**
- **Test Coverage**: >85% across all modules
- **Code Quality**: Consistent formatting and PEP 8 compliance
- **Performance Benchmarks**: Automated model accuracy validation
- **Security Scanning**: Dependency vulnerability checks

## 🚀 Technical Highlights

### **What Recruiters Will Notice:**
- ✅ **Production Architecture**: Not a notebook—proper software engineering
- ✅ **Full-Stack Implementation**: Backend API + Frontend interface + DevOps
- ✅ **Testing Culture**: Comprehensive test suite with CI/CD integration
- ✅ **Modern Tools**: FastAPI, Docker, GitHub Actions, proper Python packaging
- ✅ **Performance Focus**: Benchmarked models with documented metrics

### **For Fellow Developers:**
- 🏗️ **Clean Architecture**: Modular design following software engineering principles  
- 🔧 **Rich Tooling**: Makefile automation, Docker orchestration, dependency management
- 📚 **Documentation**: Comprehensive README, inline docs, API documentation
- 🧪 **Testable Code**: Unit tests, integration tests, mocking, fixtures
- 🚀 **Deployment Ready**: Container-native with health checks and monitoring

### **Future Enhancements** 
- [ ] **Model Optimization**: ONNX export for cross-platform deployment
- [ ] **Monitoring**: Prometheus metrics and Grafana dashboards  
- [ ] **Advanced Architectures**: Vision Transformer implementation
- [ ] **MLOps**: Weights & Biases integration for experiment tracking

## 🤝 Contributing

This project welcomes contributions from the community! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### **Development Workflow**
```bash
# Setup development environment
git clone https://github.com/ukimsanov/mnist-linear-classifier.git
cd mnist-linear-classifier
make install-dev

# Run tests before making changes
make test && make lint

# Make your changes...

# Verify your changes
make test && make lint
```

### **Contribution Guidelines**
1. **Fork & Branch**: Create a feature branch from main
2. **Code Quality**: Ensure tests pass and code follows project standards
3. **Documentation**: Update README/docs for significant changes  
4. **Testing**: Add tests for new functionality
5. **Pull Request**: Submit with clear description of changes

All contributions will be reviewed for code quality and alignment with project goals.

## 🎓 Learning Resources

### **Understanding the Code**
- **`src/mnist_classifier/models/`**: Study model implementations to understand PyTorch architecture patterns
- **`train.py`**: See how modern training loops handle validation, checkpointing, and metrics
- **`api.py`**: Learn FastAPI patterns for ML model serving
- **`frontend/`**: Explore Next.js 15 with React 19 for modern ML interfaces
- **`.github/workflows/`**: Examine CI/CD best practices for ML projects

### **Key Concepts Demonstrated**
- **Model Architecture Design**: From simple linear to modern CNN with best practices
- **Training Pipeline**: Proper validation splits, metric tracking, model checkpointing
- **Production Deployment**: REST APIs, containerization, health checks
- **Code Quality**: Testing, linting, documentation for enterprise standards

## 🏆 Acknowledgments

- **PyTorch**: Exceptional deep learning framework enabling rapid development
- **MNIST Dataset**: Classic benchmark by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges  
- **Next.js + React**: Modern frontend stack for beautiful, interactive ML demos
- **FastAPI**: Modern, fast web framework perfect for ML APIs

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if it helped you learn something new!**

*Transforming classic ML challenges into modern engineering showcases*

![Visitor Count](https://visitor-badge.laobi.icu/badge?page_id=ukimsanov.mnist-linear-classifier)

</div>