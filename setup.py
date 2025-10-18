from setuptools import setup, find_packages

setup(
    name="c2ae-reproduction",
    version="1.0.0",
    description="Reproduction of C2AE: Class Conditioned Auto-Encoder for Open-Set Recognition",
    author="Reproduction Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tensorboard>=2.12.0",
        "protobuf<=3.20.3",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "mypy>=1.2.0",
            "jupyter>=1.0.0",
        ]
    }
)
