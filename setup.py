"""Setup configuration for MNIST Classifier package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mnist-classifier",
    version="0.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive MNIST digit classifier with modern ML engineering practices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mnist-classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "Pillow>=9.3.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "pandas>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "web": [
            "gradio>=4.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.23.0",
            "pydantic>=2.0.0",
        ],
        "mlops": [
            "mlflow>=2.5.0",
            "tensorboard>=2.11.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mnist-train=mnist_classifier.train:main",
            "mnist-infer=mnist_classifier.inference:main",
            "mnist-demo=mnist_classifier.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)