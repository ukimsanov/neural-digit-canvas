<div align="center">

# �� Neural Digit Canvas

### *From hand-drawn sketches to AI predictions in milliseconds*

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-neural--digit--canvas-blue?style=for-the-badge)](https://neural-digit-canvas.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![AWS Lambda](https://img.shields.io/badge/AWS_Lambda-Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/lambda/)
[![Next.js](https://img.shields.io/badge/Next.js-15-000000?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[**🎨 Try it Live**](https://neural-digit-canvas.vercel.app) • [**📡 API Docs**](https://yohvwh25qifzy22ny6o3qtnp5i0vmmvg.lambda-url.us-east-1.on.aws/docs) • [**🤝 Contribute**](#-contributing)

</div>

---

## ✨ What is This?

A **production-ready ML project** that turns MNIST digit classification into a full-stack AI experience. Draw a digit, watch the neural network think, get instant predictions. Built with modern tech, deployed serverless, designed to impress.

**Not just another tutorial** — this is enterprise-grade ML engineering:
- 🧠 **Dual Architecture**: Linear (92.4%) & CNN (98.2%) classifiers
- 🌐 **Interactive Frontend**: Next.js 15 with beautiful animations
- ☁️ **Serverless Backend**: AWS Lambda + FastAPI
- 📦 **Container Native**: Docker + ECR deployment
- 🔄 **CI/CD Pipeline**: Automated testing & deployment

<div align="center">

### 🚀 [**Try the Live Demo**](https://neural-digit-canvas.vercel.app)

*Draw → Predict → Marvel*

</div>

---

## 🎯 Quick Start

```bash
# Clone & run in 30 seconds
git clone https://github.com/ukimsanov/neural-digit-canvas.git
cd mnist-linear-classifier

# One command to rule them all
make install && make train && make run

# 🌐 Frontend: http://localhost:3000
# 📡 API: http://localhost:8000/docs
```

**Or use Docker:**
```bash
docker-compose up --build
```

That's it. No complicated setup, no 50-step tutorials. Just ML that works.

---

## 🧠 Architecture

### Models
| Model | Accuracy | Params | Speed | Use Case |
|-------|----------|--------|-------|----------|
| **Linear** | 92.4% | 7.8K | ~2ms | Edge/Baseline |
| **CNN** | 98.2% | 102K | ~8ms | Production |

### Stack
```
Frontend (Next.js 15 + React 19)
          ↓
    FastAPI + Mangum
          ↓
   AWS Lambda (3GB, 180s timeout)
          ↓
  PyTorch CPU Models (~700MB)
```

**Tech Highlights:**
- 🎨 Modern UI with glassmorphism & animations
- ⚡ Serverless cold starts: 60-100s (first request), <1s (warm)
- 🔒 CORS configured for cross-origin requests
- 📊 Real-time predictions with confidence scores

---

## 📁 Project Structure

```
mnist-linear-classifier/
├── 🧠 src/mnist_classifier/    # Core ML package
│   ├── models/                 # Linear & CNN architectures
│   └── utils/                  # Training, evaluation, viz
├── 🎨 frontend/                # Next.js 15 application
│   ├── src/components/         # React components
│   └── src/app/                # App router pages
├── 📡 api.py                   # FastAPI + Mangum handler
├── 🐳 Dockerfile               # Lambda container image
├── 🎯 train.py                 # Model training CLI
├── ⚙️ Makefile                 # Dev automation
└── 🔄 .github/workflows/       # CI/CD pipelines
```

---

## 🔥 Features

### For ML Engineers
- ✅ Clean, modular PyTorch implementations
- ✅ Proper data pipelines with transforms
- ✅ Training metrics & visualization
- ✅ Model checkpointing & evaluation
- ✅ Confusion matrices & performance plots

### For Backend Developers
- ✅ FastAPI with async endpoints
- ✅ AWS Lambda containerized deployment
- ✅ OpenAPI/Swagger documentation
- ✅ Error handling & validation
- ✅ Lazy model loading for cold starts

### For Frontend Developers
- ✅ Modern Next.js 15 + TypeScript
- ✅ Interactive canvas with brush controls
- ✅ Beautiful animations & transitions
- ✅ Responsive design (mobile/desktop)
- ✅ Real-time API status indicator

### For DevOps Engineers
- ✅ Docker multi-stage builds
- ✅ ECR repository management
- ✅ Lambda Function URL + CORS
- ✅ GitHub Actions CI/CD
- ✅ Makefile automation

---

## 🌐 Deployment

### AWS Lambda (Current)
```bash
# Build for Lambda
docker build --platform linux/amd64 --provenance=false -t mnist-classifier:v8 .

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-ecr-repo>
docker tag mnist-classifier:v8 <your-ecr-repo>:v8
docker push <your-ecr-repo>:v8

# Update Lambda function
aws lambda update-function-code --function-name mnist-classifier-api --image-uri <your-ecr-repo>:v8
```

**⚠️ Cold Start Note**: First request after inactivity takes 60-100s (PyTorch image download + initialization). Subsequent requests are <1s.

**💡 Pro Tip**: Use [Provisioned Concurrency](https://docs.aws.amazon.com/lambda/latest/dg/provisioned-concurrency.html) for production to avoid cold starts (~$3-5/month).

---

## 🤝 Contributing

**We welcome contributions!** Whether you're:
- 🐛 Fixing bugs
- ✨ Adding features
- 📝 Improving docs
- 🎨 Enhancing UI/UX
- ⚡ Optimizing performance

### How to Contribute

1. **Fork** this repo
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to your branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

**Ideas for contribution:**
- [ ] Add new model architectures (ResNet, Vision Transformer)
- [ ] Implement ONNX export for cross-platform deployment
- [ ] Add model quantization for faster inference
- [ ] Create Prometheus metrics endpoint
- [ ] Build a mobile app (React Native)
- [ ] Add support for other datasets (Fashion-MNIST, CIFAR-10)
- [ ] Implement A/B testing for models
- [ ] Create Terraform/Pulumi IaC templates

**Check out [Issues](https://github.com/ukimsanov/neural-digit-canvas/issues) for tasks to work on!**

---

## 📊 Performance

### Model Metrics
```
CNN Classifier:
├── Test Accuracy: 98.16%
├── Training Time: ~2 min (10 epochs)
├── Model Size: 400KB
└── Inference: ~8ms/image

Linear Classifier:
├── Test Accuracy: 92.42%
├── Training Time: ~30 sec (5 epochs)
├── Model Size: 31KB
└── Inference: ~2ms/image
```

### API Performance
- **Cold Start**: 60-100 seconds (Lambda + PyTorch init)
- **Warm Start**: <1 second
- **Throughput**: ~125 req/sec (warm)
- **Memory**: 3GB Lambda allocation

---

## 📚 Learn More

This project demonstrates:
- 🏗️ **ML System Design**: From research to production
- 🔧 **Software Engineering**: Clean code, testing, CI/CD
- ☁️ **Cloud Architecture**: Serverless deployment patterns
- 🎨 **Full-Stack Dev**: React + FastAPI integration
- 📦 **Containerization**: Docker best practices
- 🚀 **DevOps**: Automated workflows & deployments

**Perfect for:**
- ML engineers learning production deployment
- Backend devs exploring AI/ML integration
- Frontend devs building ML interfaces
- Students showcasing portfolio projects
- Anyone curious about modern ML engineering

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

### ⭐ Star this repo if you found it helpful!

**Built with ❤️ by [Ular Kimsanov](https://github.com/ukimsanov)**

[🌐 Live Demo](https://neural-digit-canvas.vercel.app) • [📡 API](https://yohvwh25qifzy22ny6o3qtnp5i0vmmvg.lambda-url.us-east-1.on.aws/docs) • [🐛 Issues](https://github.com/ukimsanov/neural-digit-canvas/issues) • [🤝 Contribute](#-contributing)

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=ukimsanov.neural-digit-canvas)

</div>
