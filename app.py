#!/usr/bin/env python3
"""Interactive Gradio demo for MNIST classifier."""

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.mnist_classifier.models import LinearClassifier, CNNClassifier
from torchvision import transforms


class MNISTDemo:
    """Gradio demo for MNIST classifier."""

    def __init__(self, model_path: str = None, model_type: str = 'cnn'):
        """Initialize the demo.

        Args:
            model_path: Path to trained model weights
            model_type: Type of model ('linear' or 'cnn')
        """
        self.model_type = model_type
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def load_model(self, model_path: str = None) -> torch.nn.Module:
        """Load model weights."""
        if self.model_type == 'linear':
            model = LinearClassifier()
        else:
            model = CNNClassifier()

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            # Check if it's a full checkpoint or just state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from checkpoint: {model_path}")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model weights found, using random initialization")

        model.eval()
        return model

    def preprocess_drawing(self, image):
        """Preprocess drawn image for inference."""
        if image is None:
            return None
        
        print(f"DEBUG: Input type: {type(image)}")
        
        # Handle Gradio 5.x dictionary format
        if isinstance(image, dict):
            print(f"DEBUG: Dict keys: {list(image.keys())}")
            
            # Try to get the composite image first
            if 'composite' in image and image['composite'] is not None:
                extracted_image = image['composite']
                print("DEBUG: Using composite")
            # If no composite, try background
            elif 'background' in image and image['background'] is not None:
                extracted_image = image['background']
                print("DEBUG: Using background")
            else:
                print("DEBUG: No valid image data found")
                return None
                
            # Handle different image formats
            if isinstance(extracted_image, np.ndarray):
                print(f"DEBUG: Converting numpy array of shape: {extracted_image.shape}")
                # Handle different array formats
                if len(extracted_image.shape) == 3:
                    if extracted_image.shape[2] == 4:  # RGBA
                        image = Image.fromarray(extracted_image, 'RGBA').convert('RGB')
                    else:  # RGB
                        image = Image.fromarray(extracted_image, 'RGB')
                else:
                    print(f"DEBUG: Unexpected array shape: {extracted_image.shape}")
                    return None
            elif isinstance(extracted_image, Image.Image):
                print("DEBUG: Got PIL Image from composite")
                image = extracted_image
            else:
                print(f"DEBUG: Unexpected image type: {type(extracted_image)}")
                return None
                
        elif isinstance(image, Image.Image):
            print("DEBUG: Got PIL Image directly")
        else:
            print(f"DEBUG: Unexpected input type: {type(image)}")
            return None

        # Convert to grayscale
        image = image.convert('L')

        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # Invert if necessary (MNIST has white digits on black background)
        image_array = np.array(image)
        print(f"DEBUG: Image mean before inversion: {np.mean(image_array)}")
        
        # For MNIST, we want white digits on black background
        # If the background is mostly white, invert to get black background
        if np.mean(image_array) > 127:
            image = ImageOps.invert(image)
            print("DEBUG: Image inverted")

        # Save debug image to see what we're actually processing
        image.save("/tmp/debug_processed.png")
        print("DEBUG: Saved processed image to /tmp/debug_processed.png")

        return image

    def predict(self, drawing):
        """Make prediction on the drawn digit"""
        if drawing is None:
            return None, "**🎨 Canvas is empty!**\n\nDraw a digit (0-9) to see the AI prediction.", None

        try:
            # Process the drawing
            processed_image_pil = self.preprocess_drawing(drawing)
            
            if processed_image_pil is None:
                return None, "**🎨 Canvas is empty!**\n\nDraw a digit (0-9) to see the AI prediction.", None
            
            # Convert PIL image to tensor for model input
            processed_tensor = torch.tensor(np.array(processed_image_pil), dtype=torch.float32) / 255.0
            
            # Apply the same normalization used during training: mean=0.1307, std=0.3081
            processed_tensor = (processed_tensor - 0.1307) / 0.3081
            
            print(f"DEBUG: Tensor shape before unsqueeze: {processed_tensor.shape}")
            print(f"DEBUG: Tensor min/max/mean: {processed_tensor.min():.3f}/{processed_tensor.max():.3f}/{processed_tensor.mean():.3f}")
            print(f"DEBUG: Tensor sum: {processed_tensor.sum():.3f}")
            
            # Check if processed image is mostly empty (adjust threshold for normalized values)
            if abs(processed_tensor.sum()) < 1.0:  # Very little content drawn
                return None, "**✏️ Draw something more visible!**\n\nTry drawing with bolder strokes.", None

            # Add batch and channel dimensions: (H, W) -> (1, 1, H, W)
            processed_tensor = processed_tensor.unsqueeze(0).unsqueeze(0)
            print(f"DEBUG: Final tensor shape: {processed_tensor.shape}")

            # Make prediction
            with torch.no_grad():
                outputs = self.model(processed_tensor)
                print(f"DEBUG: Model outputs: {outputs}")
                probabilities = torch.softmax(outputs, dim=1).squeeze()
                print(f"DEBUG: Probabilities: {probabilities}")

            # Get predicted class and confidence
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            print(f"DEBUG: Predicted class: {predicted_class}")
            print(f"DEBUG: Confidence: {confidence}")

        except Exception as e:
            return None, f"**❌ Error:** {str(e)}\n\nTry drawing again or clearing the canvas.", None

        # Create modern, minimal visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.patch.set_facecolor('white')

        # Plot probability distribution with modern styling
        classes = list(range(10))
        probs = probabilities.numpy()
        
        # Use a modern, minimal color scheme
        primary_color = '#3b82f6'  # Blue
        secondary_color = '#e5e7eb'  # Light gray
        accent_color = '#1e40af'    # Dark blue
        
        colors = [primary_color if i == predicted_class else secondary_color for i in classes]
        
        bars = ax.bar(classes, probs, color=colors, alpha=0.9, 
                     edgecolor='white', linewidth=2, width=0.7)
        
        # Highlight the predicted class with accent color
        bars[predicted_class].set_color(accent_color)
        bars[predicted_class].set_alpha(1.0)
        
        # Clean, minimal styling
        ax.set_xlabel('Digit', fontsize=13, fontweight='500', color='#374151')
        ax.set_ylabel('Confidence', fontsize=13, fontweight='500', color='#374151')
        ax.set_title('Prediction Confidence', fontsize=16, fontweight='600', 
                    color='#111827', pad=20)
        ax.set_xticks(classes)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        
        # Add percentage labels only for significant probabilities
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            height = bar.get_height()
            if prob > 0.02:  # Only show labels for probabilities > 2%
                label_color = 'white' if i == predicted_class else '#374151'
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.1%}',
                        ha='center', va='bottom',
                        fontsize=11, 
                        fontweight='600' if i == predicted_class else '400',
                        color=label_color)
        
        plt.tight_layout()
        
        # Create clean, minimal result text
        status_indicator = "�" if confidence > 0.8 else "�" if confidence > 0.6 else "🔴"
        
        result_text = f"## {status_indicator} Predicted Digit: **{predicted_class}**\n"
        result_text += f"**Confidence:** {confidence:.1%}\n\n"
        
        # Add confidence interpretation
        if confidence > 0.9:
            result_text += "*Excellent prediction with very high confidence*\n\n"
        elif confidence > 0.7:
            result_text += "*Good prediction with solid confidence*\n\n"
        elif confidence > 0.5:
            result_text += "*Moderate confidence - try drawing clearer*\n\n"
        else:
            result_text += "*Low confidence - digit may be unclear*\n\n"
        
        # Show top 3 predictions in a clean format
        result_text += "**Top Predictions:**\n"
        top3_indices = torch.topk(probabilities, 3).indices
        
        for i, idx in enumerate(top3_indices):
            confidence_bar = "█" * int(probabilities[idx].item() * 10) + "░" * (10 - int(probabilities[idx].item() * 10))
            result_text += f"**{idx.item()}:** {probabilities[idx].item():.1%} `{confidence_bar}`\n"
        
        return fig, result_text, processed_image_pil

    def clear_all(self):
        """Clear all inputs and outputs."""
        return None, None, "", None


def create_demo():
    """Create the Gradio demo interface"""
    # Use the trained CNN model
    model_path = Path(__file__).parent / "outputs" / "cnn" / "final_model.pth"
    demo_model = MNISTDemo(model_path=str(model_path), model_type='cnn')
    
    # Modern minimalistic CSS with better contrast
    custom_css = """
    /* Global styles */
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background: #f8fafc !important;  /* Solid light background */
        color: #1e293b !important;  /* Ensure dark text globally */
    }

    /* Force only text elements to be dark, not canvas/images */
    p, span, h1, h2, h3, h4, h5, h6, label, a, li, td, th {
        color: #1e293b !important;
    }

    /* More specific for divs to avoid affecting containers */
    div.prose, div.markdown-text, div[class*="text"] {
        color: #1e293b !important;
    }

    /* Ensure canvas and image areas have proper backgrounds */
    canvas {
        background-color: white !important;
    }

    /* Sketchpad specific styling */
    #drawing-canvas canvas {
        background-color: white !important;
    }

    #drawing-canvas {
        background-color: white !important;
    }

    /* Ensure image containers are white */
    .image-container, img {
        background-color: white !important;
    }

    /* Keep button text white */
    .primary-button {
        color: white !important;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2.5rem 2rem;
        background: #ffffff !important;  /* Solid white background */
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid #cbd5e1;
    }
    
    .main-title {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #1e293b !important;  /* Solid dark color instead of gradient */
        margin-bottom: 0.8rem !important;
        letter-spacing: -0.025em !important;
    }

    /* Ensure all markdown headings are dark */
    .main-title h1 {
        color: #1e293b !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
    }
    
    .main-subtitle {
        font-size: 1.1rem !important;
        color: #475569 !important;
        font-weight: 500 !important;
    }

    /* Fix all markdown text colors */
    .markdown-text, .prose {
        color: #1e293b !important;
    }

    .markdown-text p, .prose p {
        color: #475569 !important;
    }
    
    /* Drawing section */
    .drawing-section {
        background: #ffffff !important;  /* Solid white background */
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08) !important;
        border: 1px solid #cbd5e1 !important;
        height: fit-content !important;
    }
    
    .section-title {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
        margin-bottom: 1.5rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    /* Ensure all heading text is dark */
    .section-title h3, h3 {
        color: #1e293b !important;
        font-weight: 700 !important;
    }
    
    /* Results section */
    .results-section {
        background: #ffffff !important;  /* Solid white background */
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08) !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Buttons */
    .primary-button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3) !important;
    }
    
    .primary-button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4) !important;
    }
    
    .secondary-button {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 1rem 2rem !important;
        color: #1e293b !important;
        transition: all 0.3s ease !important;
    }
    
    .secondary-button:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%) !important;
        border-color: #64748b !important;
        transform: translateY(-1px) !important;
    }
    
    /* Accordion styles */
    .accordion-header {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
        border: 1px solid #94a3b8 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    
    /* Results text */
    .prediction-result {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #38bdf8;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        color: #0c4a6e;
        font-weight: 500;
    }
    
    /* Footer */
    .footer-section {
        background: #ffffff !important;  /* Solid white background */
        border-radius: 20px;
        padding: 2.5rem;
        margin-top: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        border: 1px solid #cbd5e1;
        text-align: center;
    }
    
    /* Remove default gradio styling */
    .gradio-container .prose {
        max-width: none !important;
    }
    
    /* Canvas styling - Use ID for better specificity */
    #drawing-canvas, .canvas-container {
        border: 3px solid #3b82f6 !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.2) !important;
        background-color: white !important;
    }

    /* Ensure Gradio image components have white backgrounds */
    .gr-image, .gr-plot, .gr-sketchpad {
        background-color: white !important;
    }

    /* Fix for plot areas */
    .plot-container {
        background-color: white !important;
    }
    
    /* Tips and model content */
    .tips-content, .model-content {
        font-size: 1rem;
        line-height: 1.7;
        color: #334155;
        font-weight: 400;
    }
    
    .tips-content strong, .model-content strong {
        color: #0f172a;
        font-weight: 700;
    }
    """

    with gr.Blocks(
        title="MNIST Digit Classifier",
        analytics_enabled=False,
        css=custom_css,
        theme=gr.themes.Base()  # Use Base theme for better stability
    ) as demo:
        # Clean modern header
        with gr.Row(elem_classes=["main-header"]):
            with gr.Column():
                gr.Markdown(
                    "# MNIST Digit Classifier", 
                    elem_classes=["main-title"]
                )
                gr.Markdown(
                    "Draw any digit and watch our neural network predict it in real-time",
                    elem_classes=["main-subtitle"]
                )

        with gr.Row(equal_height=True):
            # Left: Drawing section
            with gr.Column(scale=1, elem_classes=["drawing-section"]):
                gr.Markdown("### Draw a Digit", elem_classes=["section-title"])
                
                canvas = gr.Sketchpad(
                    label="",
                    type="pil",
                    height=280,
                    width=280,
                    canvas_size=(280, 280),
                    image_mode='L',  # Grayscale for MNIST
                    brush=gr.Brush(default_size=15, colors=['#FFFFFF']),
                    elem_id="drawing-canvas",  # Use elem_id for better CSS targeting
                    elem_classes=["canvas-container"]
                )

                with gr.Row():
                    predict_btn = gr.Button(
                        "Predict", 
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-button"]
                    )
                    clear_btn = gr.Button(
                        "Clear", 
                        variant="secondary",
                        size="lg",
                        elem_classes=["secondary-button"]
                    )

            # Right: Results section
            with gr.Column(scale=1, elem_classes=["results-section"]):
                gr.Markdown("### Prediction Results", elem_classes=["section-title"])
                
                result_text = gr.Markdown(
                    "Draw a digit and click **Predict** to see the results...",
                    elem_classes=["prediction-result"]
                )
                
                with gr.Row():
                    result_plot = gr.Plot(
                        label="Confidence Distribution",
                        show_label=False
                    )
                    processed_img = gr.Image(
                        label="Processed Input",
                        type="pil",
                        height=140,
                        width=140,
                        show_label=True
                    )

        # Footer with collapsible info
        with gr.Row(elem_classes=["footer-section"]):
            with gr.Column():
                with gr.Accordion("Tips & Model Information", open=False, elem_classes=["accordion-header"]):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                """
                                **Drawing Tips:**
                                • Draw digits large and bold
                                • Center your drawing
                                • Use single, continuous strokes  
                                • Keep digits simple and clear
                                • Works great on mobile devices
                                """,
                                elem_classes=["tips-content"]
                            )
                        with gr.Column():
                            gr.Markdown(
                                """
                                **Model Information:**
                                • CNN with 102,026 parameters
                                • 99.2% accuracy on MNIST dataset
                                • Trained on 60,000 handwritten digits
                                • Real-time inference (~10ms)
                                • Input: 28×28 grayscale images
                                """,
                                elem_classes=["model-content"]
                            )

        # Interactive examples section
        gr.Markdown("### 🎨 Try These Examples")
        gr.Markdown("Click any example below to load it into the canvas!")
        
        # Event handlers
        predict_btn.click(
            fn=demo_model.predict,
            inputs=canvas,
            outputs=[result_plot, result_text, processed_img]
        )

        clear_btn.click(
            fn=demo_model.clear_all,
            inputs=[],
            outputs=[canvas, result_plot, result_text, processed_img]
        )
        
        # Auto-predict on canvas change for real-time feedback
        canvas.change(
            fn=demo_model.predict,
            inputs=canvas,
            outputs=[result_plot, result_text, processed_img],
            show_progress=False
        )

        # Footer with enhanced styling
        gr.HTML("""
            <div style="text-align: center; margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
                <h3>🚀 About This Demo</h3>
                <p>This interactive demo showcases a state-of-the-art CNN trained on the MNIST dataset, achieving >99% accuracy.</p>
                <p><strong>✨ Features:</strong> Real-time prediction • Confidence visualization • Mobile-friendly drawing • Advanced preprocessing</p>
                <p>
                    <a href="https://github.com/yourusername/mnist-classifier" style="color: #FFE4E1; text-decoration: none;">📖 View on GitHub</a> | 
                    <a href="https://github.com/yourusername/mnist-classifier/issues" style="color: #FFE4E1; text-decoration: none;">🐛 Report Issues</a>
                </p>
            </div>
        """)

    return demo


if __name__ == "__main__":
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Create public link
        show_error=True
    )