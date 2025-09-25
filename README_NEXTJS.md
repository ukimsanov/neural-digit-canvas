# 🎯 MNIST Digit Classifier - Modern Next.js Interface

A production-grade MNIST digit classifier with a modern **Next.js + TypeScript** frontend and **FastAPI** backend. Draw digits and watch neural networks predict in real-time!

![Next.js](https://img.shields.io/badge/Next.js-15.5-black?logo=next.js)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-v4-38bdf8?logo=tailwindcss)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)

## ✨ What's New

- **Modern Frontend**: Beautiful Next.js 15.5 interface with TypeScript
- **Tailwind CSS v4**: Latest styling with improved performance
- **Real-time Predictions**: Interactive canvas with instant feedback
- **Dual Model Support**: Choose between Linear (fast) or CNN (accurate)
- **Professional UI/UX**: Clean, responsive design that works on all devices

## 🚀 Quick Start

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/ukimsanov/mnist-linear-classifier.git
cd mnist-linear-classifier

# Run everything with one command
./run.sh  # Mac/Linux
# or
run.bat   # Windows
```

That's it! The application will:
- ✅ Set up Python virtual environment
- ✅ Install all dependencies
- ✅ Start the FastAPI backend
- ✅ Launch the Next.js frontend

**Access the app at:**
- 🌐 **Frontend**: http://localhost:3000
- 📡 **API Docs**: http://localhost:8000/docs

## 🛠️ Manual Setup

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### Step 1: Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Train the models (optional - pre-trained models included)
python train.py --model both --epochs 10
```

### Step 2: Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Return to root
cd ..
```

### Step 3: Run the Application

**Option 1: Using Make (recommended)**
```bash
make run  # Runs both services
```

**Option 2: Run services separately**

Terminal 1 - API:
```bash
source venv/bin/activate
python api.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

## 🐳 Docker Deployment

```bash
# Build and run all services
docker-compose up --build

# Or run specific services
docker-compose up api frontend
```

Services will be available at:
- Frontend: http://localhost:3000
- API: http://localhost:8000

## 📁 Project Structure

```
mnist-linear-classifier/
├── frontend/                    # Next.js application
│   ├── src/
│   │   ├── app/                # Next.js app router
│   │   ├── components/         # React components
│   │   │   ├── DrawingCanvas.tsx
│   │   │   ├── PredictionResult.tsx
│   │   │   └── ModelSelector.tsx
│   │   └── lib/                # Utilities and API client
│   ├── package.json
│   └── next.config.ts
├── src/mnist_classifier/        # Python ML package
│   ├── models/                 # Neural network architectures
│   │   ├── linear.py          # Linear classifier
│   │   └── cnn.py             # CNN architecture
│   └── utils/                  # ML utilities
├── api.py                      # FastAPI backend
├── train.py                    # Model training script
├── docker-compose.yml          # Docker orchestration
├── run.sh                      # Startup script (Mac/Linux)
└── run.bat                     # Startup script (Windows)
```

## 🎮 Features

### Interactive Drawing Canvas
- Touch-friendly drawing interface
- Real-time preprocessing
- Clear and predict buttons
- Mobile responsive

### Model Selection
- **Linear Classifier**: 92.4% accuracy, ~2ms inference
- **CNN**: 98.2% accuracy, ~8ms inference
- Switch models on-the-fly

### Confidence Visualization
- Beautiful bar charts showing prediction confidence
- Top-3 predictions with probabilities
- Color-coded confidence levels

### API Integration
- RESTful API with FastAPI
- OpenAPI documentation
- Real-time health monitoring
- CORS enabled for frontend communication

## 🧪 Development

### Frontend Development

```bash
cd frontend
npm run dev     # Development server with hot reload
npm run build   # Production build
npm run lint    # Run ESLint
```

### Backend Development

```bash
# Run API with auto-reload
uvicorn api:app --reload

# Run tests
pytest tests/

# Train models
python train.py --model both --epochs 10
```

### Available Make Commands

```bash
make help           # Show all commands
make run            # Run both services
make frontend       # Run frontend only
make api           # Run API only
make train          # Train models
make test          # Run tests
make docker-up     # Start Docker services
make docker-down   # Stop Docker services
```

## 🔧 Configuration

### Environment Variables

Create `.env` file in the frontend directory:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### API Configuration

The API runs on port 8000 by default. Modify in `api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Frontend Configuration

Next.js runs on port 3000 by default. Modify in `package.json`:
```json
"dev": "next dev -p 3001"  // Change port
```

## 📊 Model Performance

| Model | Accuracy | Parameters | Inference Time | Size |
|-------|----------|------------|----------------|------|
| Linear | 92.42% | 7,850 | ~2ms | 31KB |
| CNN | 98.16% | 102,026 | ~8ms | 400KB |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Next.js**: Modern React framework
- **Tailwind CSS v4**: Utility-first CSS framework
- **FastAPI**: High-performance Python API framework
- **PyTorch**: Deep learning framework
- **MNIST Dataset**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
