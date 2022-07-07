FROM ubuntu:18.04

# Install dependencies.
# g++ (v. 5.4) does not work: https://github.com/tensorflow/tensorflow/issues/13308
RUN apt-get update && apt-get install -y \
    curl \
    zip \
    unzip \
    unrar \
    software-properties-common \
    pkg-config \
    g++-4.8 \
    zlib1g-dev \
    python \
    lua5.1 \
    liblua5.1-0-dev \
    libffi-dev \
    gettext \
    freeglut3 \
    libsdl2-dev \
    libosmesa6-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    python-dev \
    build-essential \
    git \
    python-setuptools \
    python-pip \
    libjpeg-dev \
    python3-pip \
    python3 \
    ffmpeg \
    xvfb \
    wget

# Install bazel
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | \
    apt-key add - && \
    apt-get update && apt-get install -y bazel

RUN pip3 install --upgrade pip

# Install TensorFlow and other dependencies
RUN pip3 install tensorflow==1.9.0 dm-sonnet==1.23 gym[atari]==0.15.7 opencv-python

WORKDIR scalable_agent
COPY *.py *.cc ./

# Build dynamic batching module.
RUN TF_INC="$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')" && \
    TF_LIB="$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')" && \
    g++-4.8 -std=c++11 -shared batcher.cc -o batcher.so -fPIC -I $TF_INC -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -L$TF_LIB -ltensorflow_framework

# Run tests.
# RUN python3 py_process_test.py
# RUN python3 dynamic_batching_test.py
# RUN python3 vtrace_test.py

WORKDIR /home/

# Install ROMS
RUN wget http://www.atarimania.com/roms/Roms.rar
RUN unrar -o+ e Roms.rar ROMS/
RUN python3 -m atari_py.import_roms ROMS

# Run.
CMD ["sh", "-c", "python3 experiment.py --total_environment_frames=10000 && python experiment.py --mode=test --test_num_episodes=5"]

# Docker commands:
#   docker rm scalable_agent -v
#   docker build -t scalable_agent .
#   docker run --name scalable_agent scalable_agent
