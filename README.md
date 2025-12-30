# Efficient Memory Management in 3D Gaussian Splatting for Real-Time Rendering

3D Gaussian Splatting (3DGS) has become a popular method for efficient and high-quality novel view synthesis. However, its training process demands excessive GPU memory, primarily due to uniform storage of high-order spherical harmonic coefficients and rapid increases in the number of Gaussians. To address this, we introduce a memory-efficient approach that dynamically adjusts spherical harmonic orders based on the expressive needs of each Gaussian and employs a saliency-based pruning strategy to remove redundant Gaussians. Our method reduces peak GPU memory usage by up to 32% while maintaining rendering quality comparable to the original approach. Experiments on benchmark datasets demonstrate the effectiveness of our approach.


## Setup

### Local Setup

Our default, experiments were conducted on Ubuntu22.04 and cuda11.8
```shell
git clone https://github.com/Allen-key-20/Efficient-Memory-Management-in-3D-Gaussian-Splatting.git --init --recursive
cd Efficient-Memory-Management-in-3D-Gaussian-Splatting

conda env create -n name python=3.10 -y
conda activate name

pip install ./torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl
pip install ./torchaudio-2.0.2+cu118-cp310-cp310-linux_x86_64.whl
pip install ./torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl

pip install -r requirements.txt

pip install ./our-diff-gaussian-rasterization
pip install ./fused-ssim
.pip install ./simple-knn

```
### Dataset
You can find three datasets in [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) . Put the downloaded dataset into the "data" folder

```
data
  ├── Mip-NeRF360
  │   ├── bicycle
  │   │     ├── images
  │   │     └── sparse
  │   │     ···
  ├── Deep Blending
  │   ├── playroom
  │   │     ├── images
  │   │     └── sparse
  │   │     ···
  ├── Tanks&Temples
  │   ├── truck
  │   │     ├── images
  │   │     └── sparse
  │   │     ···
```

### Running

Train a single scene. Set `scene = ' '` in the `run.py` file to the name of the scenario to be trained. And set `dateset = ' '` to which the scene belongs.
```shell
python run.py 
```

Train the entire dataset. Train the default three datasets.
```shell
python run_folder.py 
```
### Note

Our code is modified based on [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) . The simpleknn in the submodules folder is also from this project.

