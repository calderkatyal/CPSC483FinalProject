# Use the latest PyTorch CPU image
FROM pytorch/pytorch:latest

# Set the working directory inside the container
WORKDIR /workspace

# Install any necessary system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    dgl==1.1.3 \
    joblib==1.3.2 \
    numpy==1.26.4 \
    ogb==1.3.6 \
    scikit-learn==1.3.2 \
    scipy==1.12.0 \
    torch_sparse==0.6.18 \
    tqdm==4.66.1

# Optional: Copy your Python scripts (model.py, utils.py, etc.) into the container
COPY . /workspace

# Set the entrypoint for the container
ENTRYPOINT ["/bin/bash"]

