FROM nvidia/cuda:12.3.0-devel-ubuntu22.04
ARG USER_ID=1576554604
ARG GROUP_ID=1000
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages including OpenCTM tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git wget unzip bzip2 sudo build-essential \
        ca-certificates openssh-server vim ffmpeg \
        libsm6 libxext6 python3-opencv gcc-11 g++-11 cmake \
        libboost-dev libpython3-dev pybind11-dev \
        zlib1g-dev openctm-tools libopenctm-dev \
        xauth x11-apps swig docker.io

# conda
ENV PATH=/opt/conda/bin:$PATH 
RUN wget --quiet \
    https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm -rf /tmp/*

# Install FAISS using conda (place this after the conda installation)
RUN /opt/conda/bin/conda install -y -c conda-forge python=3.10 faiss-gpu==1.7.4

# Create the user
RUN addgroup --gid $GROUP_ID user
RUN useradd -l --create-home -s /bin/bash --uid $USER_ID --gid $GROUP_ID docker
RUN adduser docker sudo

# Add user to the docker group (so it can run docker commands)
RUN usermod -aG docker docker


RUN echo "docker ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER docker

# Setup hierarchical_3d_gaussians
RUN /opt/conda/bin/python -m ensurepip
RUN /opt/conda/bin/python -m pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
RUN /opt/conda/bin/python -m pip install plyfile tqdm joblib exif scikit-learn timm==0.4.5 opencv-python==4.9.0.80 gradio_imageslider gradio==4.29.0 matplotlib pyproj Pillow piexif

# Install additional Python packages
RUN /opt/conda/bin/python -m pip install laspy lazrs open3d trimesh

# Install COLMAP dependencies
RUN sudo apt-get install -y --no-install-recommends \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libmetis-dev \
    libflann-dev \
    libsqlite3-dev \
    libceres-dev

# Clone COLMAP repository
RUN sudo git clone https://github.com/colmap/colmap

RUN cd /colmap && \
    sudo mkdir build && \
    cd build && \
    sudo cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="60;75;80;86;87;90" .. && \
    sudo make -j8 && \
    sudo make install

WORKDIR /host

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the entrypoint script executable
RUN sudo chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]