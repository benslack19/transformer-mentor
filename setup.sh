#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install SSH
sudo apt-get update && sudo apt-get install -y openssh-client openssh-server

echo "--- Running make setup ---"
make setup

echo "--- Changing default shell to zsh ---"
# Make sure zsh is actually installed before trying to chsh
if command -v zsh &> /dev/null; then
    # Use sudo here to run chsh without a password if sudoers is configured
    sudo chsh -s "$(command -v zsh)" vscode || true # chsh might fail if user is already zsh, so allow it to fail
else
    echo "Zsh not found, skipping chsh."
fi

echo "--- Installing Oh My Zsh if not present ---"
if [ ! -d "${HOME}/.oh-my-zsh" ]; then
    bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" --unattended
else
    echo "Oh My Zsh already installed."
fi

echo "--- Copying .zshrc template ---"
if [ -f "/home/vscode/.zshrc.template" ]; then
    cp "/home/vscode/.zshrc.template" "/home/vscode/.zshrc"
    chown vscode:vscode "/home/vscode/.zshrc"
else
    echo "Error: /home/vscode/.zshrc.template not found. Skipping .zshrc setup."
    exit 1
fi

echo "--- Sourcing .zshrc for current session (optional, for postCreateCommand context) ---"
zsh -c 'source ~/.zshrc' || true

echo "--- Dev container setup complete! ---"
