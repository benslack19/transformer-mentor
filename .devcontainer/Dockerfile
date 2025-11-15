# Use a slim-bookworm base image for a lightweight yet stable environment
FROM python:3.12-slim-bookworm

# Set environment variables for non-interactive installs (avoids prompting for configuration input)
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        openssh-client \
        wget \
        lsof \
        cmake \
        gcc \
        g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python tools globally (no venv)
RUN python -m pip install --upgrade pip
RUN python -m pip install flake8 ipython ipykernel ipywidgets tqdm uv

# Install oh-my-zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting

# Copy dependency file
# This layer will bust the cache when you change dependencies,
# but all the slow installs above will be preserved.
COPY pyproject.toml /deps/pyproject.toml

# Set the default working directory
WORKDIR /workspaces/transformer-mentor

ENTRYPOINT []
