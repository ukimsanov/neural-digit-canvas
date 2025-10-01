#!/usr/bin/env python3
"""FastAPI service for MNIST classifier."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np
from io import BytesIO
import base64
from typing import List, Dict, Optional
from pydantic import BaseModel
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mnist_classifier.models import LinearClassifier, CNNClassifier
from torchvision import transforms


class PredictionRequest(BaseModel):
    """Request model for base64 encoded image."""
    image: str  # Base64 encoded image
    model_type: str = "cnn"
    top_k: int = 3


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: int
    confidence: float
    top_k_predictions: List[Dict[str, float]]
    model_type: str
    model_parameters: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: List[str]
    version: str


# Initialize FastAPI app
app = FastAPI(
    title="MNIST Classifier API",
    description="REST API for MNIST digit classification",
    version="0.2.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://neural-digit-canvas.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelService:
    """Service class for model management."""

    def __init__(self):
        """Initialize model service."""
        self.models = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.load_models()

    def load_models(self):
        """Load pre-trained models."""
        # Try to load CNN model
        cnn_paths = [
            "outputs/cnn/checkpoints/best_model.pth",
            "outputs/cnn/final_model.pth"
        ]
        for path in cnn_paths:
            if Path(path).exists():
                model = CNNClassifier()
                checkpoint = torch.load(path, map_location='cpu')
                # Check if it's a full checkpoint or just state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                self.models['cnn'] = model
                print(f"Loaded CNN model from {path}")
                break
        else:
            # Use untrained model as fallback
            model = CNNClassifier()
            model.eval()
            self.models['cnn'] = model
            print("Using untrained CNN model")

        # Try to load Linear model
        linear_paths = [
            "outputs/linear/checkpoints/best_model.pth",
            "outputs/linear/final_model.pth"
        ]
        for path in linear_paths:
            if Path(path).exists():
                model = LinearClassifier()
                checkpoint = torch.load(path, map_location='cpu')
                # Check if it's a full checkpoint or just state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                model.eval()
                self.models['linear'] = model
                print(f"Loaded Linear model from {path}")
                break
        else:
            # Use untrained model as fallback
            model = LinearClassifier()
            model.eval()
            self.models['linear'] = model
            print("Using untrained Linear model")

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference."""
        # Convert to grayscale
        image = image.convert('L')

        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Invert if necessary
        image_array = np.array(image)
        if np.mean(image_array) > 127:
            image = ImageOps.invert(image)

        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor

    def predict(
        self,
        image: Image.Image,
        model_type: str = "cnn",
        top_k: int = 3
    ) -> Dict:
        """Make prediction on image."""
        if model_type not in self.models:
            raise ValueError(f"Model type {model_type} not available")

        model = self.models[model_type]

        # Preprocess image
        image_tensor = self.preprocess_image(image)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1).squeeze()

        # Get top k predictions
        top_probs, top_classes = torch.topk(probabilities, min(top_k, 10))

        # Prepare response
        predicted_class = top_classes[0].item()
        confidence = top_probs[0].item()

        top_k_predictions = [
            {"class": int(c), "probability": float(p)}
            for c, p in zip(top_classes.tolist(), top_probs.tolist())
        ]

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_k_predictions": top_k_predictions,
            "model_type": model_type,
            "model_parameters": sum(p.numel() for p in model.parameters())
        }


# Initialize model service
model_service = ModelService()


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(model_service.models.keys()),
        version="0.2.0"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...), model_type: str = "cnn", top_k: int = 3):
    """Predict digit from uploaded image file."""
    try:
        # Read and open image
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # Make prediction
        result = model_service.predict(image, model_type, top_k)

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: PredictionRequest):
    """Predict digit from base64 encoded image."""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_data))

        # Make prediction
        result = model_service.predict(
            image,
            request.model_type,
            request.top_k
        )

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models."""
    models_info = []
    for name, model in model_service.models.items():
        models_info.append({
            "name": name,
            "type": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "loaded": True
        })
    return {"models": models_info}


@app.get("/model/{model_type}/info")
async def model_info(model_type: str):
    """Get detailed information about a specific model."""
    if model_type not in model_service.models:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")

    model = model_service.models[model_type]

    # Get model architecture details
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layers.append({
                "name": name,
                "type": module.__class__.__name__,
                "parameters": sum(p.numel() for p in module.parameters())
            })

    return {
        "model_type": model_type,
        "class_name": model.__class__.__name__,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "layers": layers
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)