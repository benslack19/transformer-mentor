# Use a slim-bookworm base image for a lightweight yet stable environment
FROM python:3.12-slim-bookworm

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /workspaces/transformer-mentor

# Create a non-root user for security best practices
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
        # Dependencies for pre-commit hooks and Python packages that need compilation
        libssl-dev \
        libffi-dev \
        # Install Zsh for the interactive shell
        zsh \
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