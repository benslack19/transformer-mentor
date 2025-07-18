# Use a slim-bookworm base image for a lightweight yet stable environment
FROM python:3.12-slim-bookworm

# Set environment variables for non-interactive installs (avoids prompting for configuration input)
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /workspaces/transformer-mentor

# Create a non-root user for security best practices
ARG USERNAME=vscode
# Using UID/GID 1000 is common because it often matches the default UID/GID of the first non-root user on Linux distributions
# can help prevent permission issues when bind-mounting volumes from the host.
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Each RUN instruction creates a new layer on top of the current image
# using && \ chains multiple commands which is a best practice to minimize the number of layers (image size)
# run apt-get update before apt-get install
# prevents installation of "recommended" packages that aren't strictly necessary
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-venv \
        python3-pip \
        git \
        curl \
        build-essential \
        libssl-dev \
        libffi-dev \
        zsh \
        sudo \
        openssh-client \
    && echo $USERNAME ALL=\(ALL\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy .zshrc-template and setup.sh from the local build context
# Ensure these files are present in the root of your local repository
COPY .zshrc-template /home/$USERNAME/.zshrc.template
COPY setup.sh /usr/local/bin/setup.sh

# Make setup.sh executable and set its ownership after copying
# This ensures it can be run by the non-root user.
RUN chmod +x /usr/local/bin/setup.sh && \
    chown $USERNAME:$USERNAME /usr/local/bin/setup.sh

# Switch to the non-root user for subsequent operations and runtime
USER $USERNAME

# Define the default command that will be executed when the container starts
# This ensures zsh is the default shell if the image is run directly,
# but it will be overridden by VS Code's devcontainer.json for integrated terminals.
CMD ["zsh"]
