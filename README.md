# Depth Estimation Project

This repository contains implementations of monocular depth estimation using various deep learning approaches. The project supports both single-image depth estimation and real-time depth estimation from video/webcam feeds.

## 📚 Table of Contents
1. [Installation](#installation)
2. [Features](#features)
3. [Algorithms Used](#algorithms-used)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [Training](#training)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

## Installation

### Conda Environment Setup (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/NiharP31/3D_depth.git
cd 3D_depth
```

2. Create a new Conda environment:
```bash
# Create new environment with Python 3.8
conda create -n scene python=3.8
```

3. Activate the environment:
```bash
conda activate scene
```

4. Install PyTorch with CUDA support (if available):
```bash
# For CUDA support (recommended if you have a GPU)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only
conda install pytorch torchvision cpuonly -c pytorch
```

5. Install other dependencies:
```bash
# Core dependencies
conda install opencv matplotlib pillow scipy
pip install timm

# Optional dependencies
conda install tensorboard pyyaml h5py tqdm
```

### Conda Environment Management

```bash
# List all conda environments
conda env list

# Deactivate current environment
conda deactivate

# Remove environment if needed
conda env remove -n scene

# Export environment
conda env export > environment.yml

# Create environment from yml
conda env create -f environment.yml
```

### Manual Installation (Alternative)

If not using Conda, follow these steps:

1. Create and activate a virtual environment:
```bash
# Windows
python -m venv env
.\env\Scripts\activate

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Features

- Single image depth estimation
- Real-time depth estimation from webcam
- Training capabilities with NYU Depth V2 dataset
- Multiple implementation options:
  1. Custom trained model with ResNet34 backbone
  2. MiDaS small model for single images
  3. MiDaS small model for real-time video
- GPU acceleration support
- Real-time visualization
- Frame and depth map saving capabilities

## Algorithms Used

### 1. MiDaS (Mixed Dense Architecture Search)
- **Description**: State-of-the-art model for monocular depth estimation
- **Key Features**:
  - Multi-dataset training
  - Domain adaptation
  - Scale-invariant prediction
- **Architecture**:
  - Encoder: DPT (Dense Prediction Transformer)
  - Decoder: Multi-scale feature fusion
- **Variants Used**:
  - MiDaS Small: Optimized for real-time applications
  - Resolution: 384x384 pixels

### 2. Custom ResNet-based Model
- **Architecture**:
  - Encoder: ResNet34 backbone
  - Decoder: Custom upsampling with skip connections
- **Features**:
  - Skip connections for better detail preservation
  - Batch normalization for training stability
  - Bilinear upsampling for smooth depth maps
- **Training**:
  - Loss: MSE (Mean Squared Error)
  - Optimizer: Adam
  - Learning rate: 0.001

## Project Structure

```
3D_depth/
├── densedepth.py      # Custom implementation with training
├── midas_depth.py     # Single image MiDaS implementation
├── midas_live.py      # Real-time MiDaS implementation
├── requirements.txt   # Project dependencies
├── environment.yml    # Conda environment file
└── README.md         # Project documentation
```

## Usage

Make sure your Conda environment is activated:
```bash
conda activate scene
```

### 1. Custom Implementation (densedepth.py)
```python
# For training
python densedepth.py --mode train

# For inference
python densedepth.py --mode inference
```

### 2. Single Image Depth Estimation (midas_depth.py)
```python
python midas_depth.py --image path/to/your/image.jpg
```

### 3. Real-time Depth Estimation (midas_live.py)
```python
python midas_live.py
```

### Controls
- `q`: Quit application
- `s`: Save current frame and depth map
- ESC: Exit

## Training

### Dataset Preparation
1. Download NYU Depth V2 dataset:
```bash
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
```

2. Convert dataset:
```python
python tools/prepare_nyu.py
```

### Training Process
1. Configure training parameters in code:
```python
epochs = 10
learning_rate = 0.001
batch_size = 4
```

2. Start training:
```python
python densedepth.py --mode train
```

## Troubleshooting

### Common Conda Issues

1. Package conflicts:
```bash
# Remove environment and recreate
conda deactivate
conda env remove -n scene
conda create -n scene python=3.8
conda activate scene
```

2. CUDA issues:
```bash
# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

3. OpenCV issues:
```bash
# Reinstall OpenCV
conda install -c conda-forge opencv
```

### Performance Issues

1. CUDA out of memory:
```python
# Reduce batch size
batch_size = 2  # Default: 4
```

2. Low FPS:
```python
# Reduce input resolution
self.target_size = (240, 320)  # Default: (480, 640)
```

## Customization

### Resolution
```python
self.target_size = (height, width)  # Default: (480, 640)
```

### Color Maps
Available options:
- COLORMAP_MAGMA
- COLORMAP_INFERNO
- COLORMAP_VIRIDIS
- COLORMAP_PLASMA

```python
cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)
```

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/NewFeature`
3. Commit changes: `git commit -am 'Add NewFeature'`
4. Push to branch: `git push origin feature/NewFeature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MiDaS](https://github.com/intel-isl/MiDaS) for the pre-trained models
- [NYU Depth V2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- [PyTorch](https://pytorch.org/) team
- [OpenCV](https://opencv.org/) community