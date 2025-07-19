#!/bin/bash

#te
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

echo "--- Setting up SSH agent forwarding ---"
# Ensure SSH directory exists and has correct permissions
mkdir -p /home/vscode/.ssh
chown -R vscode:vscode /home/vscode/.ssh
chmod 700 /home/vscode/.ssh

# Set correct permissions on the mounted SSH key files
if [ -f "/home/vscode/.ssh/id_ed25519" ]; then
    chmod 600 /home/vscode/.ssh/id_ed25519
    chown vscode:vscode /home/vscode/.ssh/id_ed25519
fi

if [ -f "/home/vscode/.ssh/id_ed25519.pub" ]; then
    chmod 644 /home/vscode/.ssh/id_ed25519.pub
    chown vscode:vscode /home/vscode/.ssh/id_ed25519.pub
fi

# Create a minimal SSH config for the specific key
cat > /home/vscode/.ssh/config << 'EOF'
Host *
    AddKeysToAgent yes
    ForwardAgent yes
    IdentitiesOnly yes
    IdentityFile ~/.ssh/id_ed25519
EOF
chown vscode:vscode /home/vscode/.ssh/config
chmod 600 /home/vscode/.ssh/config

# Test SSH agent forwarding
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
    echo "WARNING: SSH agent socket not found at $SSH_AUTH_SOCK"
    echo "SSH agent forwarding may not be working."
fi

echo "--- Sourcing .zshrc for current session (optional, for postCreateCommand context) ---"
zsh -c 'source ~/.zshrc' || true

echo "--- Dev container setup complete! ---"
