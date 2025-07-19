#!/bin/bash

# test
# Exit immediately if a command exits with a non-zero status.
set -e

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
    bash -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" --unattended || true
    # Added || true to the Oh My Zsh install command.
    # Sometimes it exits with a non-zero code even on success if it thinks
    # it's interactive, which can fail the 'set -e' at the top.
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

echo "--- Setting up SSH directory permissions ---"
# Ensure SSH directory exists and has correct permissions
mkdir -p /home/vscode/.ssh
chown -R vscode:vscode /home/vscode/.ssh
chmod 700 /home/vscode/.ssh

# No need to chown/chmod individual key files or create ~/.ssh/config here
# if relying on VS Code's native agent forwarding.
# VS Code will set SSH_AUTH_SOCK to its internal proxy.

# Test SSH agent forwarding (using the correct method for VS Code)
echo "--- Testing SSH agent forwarding ---"
if [ -S "$SSH_AUTH_SOCK" ]; then
    echo "SSH agent socket found at: $SSH_AUTH_SOCK"
    if ssh-add -l > /dev/null 2>&1; then
        echo "SSH agent forwarding is working! Keys available:"
        ssh-add -l
    elif [ $? -eq 1 ]; then
        echo "SSH agent is running but no keys are loaded."
        echo "You may need to run 'ssh-add' on your host machine first."
    else
        echo "SSH agent socket exists but is not responding properly."
    fi
else
    echo "WARNING: SSH_AUTH_SOCK environment variable not set or socket not found."
    echo "SSH agent forwarding may not be working through VS Code."
    echo "Verify remote.SSH.enableAgentForwarding is true in VS Code settings."
fi

echo "--- Sourcing .zshrc for current session (optional, for postCreateCommand context) ---"
zsh -c 'source ~/.zshrc' || true

echo "--- Dev container setup complete! ---"
