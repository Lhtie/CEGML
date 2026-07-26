# Use CUDA-enabled PyTorch base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Keep the native solver version explicit and reproducible.
ARG BLACK_VERSION=v26.05.0

# Set working directory
WORKDIR /app

# Install project and BLACK build/runtime dependencies.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    --no-install-recommends \
    git \
    curl \
    build-essential \
    libfmt-dev \
    libtsl-hopscotch-map-dev \
    nlohmann-json3-dev \
    libz3-dev \
    ripgrep \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# BLACK requires CMake >= 3.25. Installing CMake from PyPI avoids depending on
# the CMake version shipped by the base image's Linux distribution.
RUN python -m pip install --no-cache-dir "cmake>=3.25,<5"

# Build BLACK with the Z3 backend and install the `black` executable globally.
# Other optional SAT backends and the formula test suite are disabled to keep
# the image build deterministic and reasonably small.
RUN git clone \
        --branch "${BLACK_VERSION}" \
        --depth 1 \
        https://github.com/black-sat/black.git \
        /tmp/black-sat \
    && cmake \
        -S /tmp/black-sat \
        -B /tmp/black-sat/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DENABLE_Z3=YES \
        -DENABLE_CVC5=NO \
        -DENABLE_CMSAT=NO \
        -DENABLE_MATHSAT=NO \
        -DENABLE_MINISAT=NO \
        -DENABLE_FORMULAS_TESTS=NO \
    && cmake --build /tmp/black-sat/build --parallel \
    && cmake --install /tmp/black-sat/build \
    && rm -rf /tmp/black-sat

# Fail the image build immediately if BLACK or its Z3 backend is unusable.
RUN black -v \
    && test "$(black solve -f 'p & !p')" = "UNSAT" \
    && test "$(black solve -f 'F p')" = "SAT"

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .
