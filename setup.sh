#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Running make setup ---"
make setup

echo "--- Changing default shell to zsh ---"
# Make sure zsh is actually installed before trying to chsh
if command -v zsh &> /dev/null; then
    chsh -s "$(command -v zsh)" vscode || true # chsh might fail if user is already zsh, so allow it to fail
else
    echo "Zsh not found, skipping chsh."
fi

echo "--- Installing Oh My Zsh if not present ---"
if [ ! -d "${HOME}/.oh-my-zsh" ]; then
    # Use 'bash -c' for the curl command to ensure it runs correctly even if
    # the default shell for the script is not bash, and to handle piping.
    bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" --unattended
else
    echo "Oh My Zsh already installed."
fi

echo "--- Copying .zshrc template ---"
# Ensure the .zshrc.template exists and is readable
if [ -f "/home/vscode/.zshrc.template" ]; then
    cp "/home/vscode/.zshrc.template" "/home/vscode/.zshrc"
    chown vscode:vscode "/home/vscode/.zshrc"
else
    echo "Error: /home/vscode/.zshrc.template not found. Skipping .zshrc setup."
    exit 1 # Exit with error if template is missing
fi

echo "--- Sourcing .zshrc for current session (optional, for postCreateCommand context) ---"
# This is primarily for the shell context *during* postCreateCommand execution.
# Actual VS Code terminals will source it automatically.
zsh -c 'source ~/.zshrc' || true # Allow to fail if zsh isn't ready or .zshrc has errors

echo "--- Dev container setup complete! ---"