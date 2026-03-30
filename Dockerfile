FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Setup basic environment and timezone (prevent interactive tzdata prompts)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-venv \
    xvfb x11-xserver-utils \
    libgl1-mesa-glx libx11-6 libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Map Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Create working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download AI2-THOR/ALFWorld Binaries unconditionally
RUN xvfb-run -a alfworld-download --extra

# Copy repository source
COPY . .

# Virtual Framebuffer wrapper script for evaluating Unity Headlessly
CMD ["xvfb-run", "-a", "python", "scripts/run_alfworld_eval.py"]
