# Transformer Mentor: A Comprehensive AI/ML Development Environment

## Project Overview

Hello and welcome to **Transformer Mentor**! This spot is all about helping you learn, play around with, and build Transformer models from the ground up. Whether you're just starting your journey into NLP and deep learning, or you've been around the block a few times and want a super smooth setup, you've come to the right place! We've got a solid, repeatable environment here to get you going fast.

Our main goal? To make Transformer architectures make sense! We're doing this by mixing the "why it works" (theory) with the "how to make it work" (hands-on coding). Let's dive in! This repo is inspired by the idea that teaching is an opportunity to gain a deeper understanding of a topic while helping others learn as well.

## Features & Technologies

This project leverages cutting-edge tools and best practices to provide a seamless development experience:

* **Transformer Models:** Focus on understanding and implementing various Transformer architectures (e.g., Attention mechanisms, Encoders, Decoders).

* **PyTorch:** The primary deep learning framework for building and training models.

* **Hugging Face Ecosystem:** Utilize `transformers` and `datasets` libraries for easy access to pre-trained models and large datasets.

* **Jupyter Lab:** Interactive development environment for experimentation, prototyping, and visualization within `notebooks/`.

* **VS Code Dev Containers:** A fully isolated and reproducible development environment, ensuring consistent setup across different machines and collaborators.

* **Zsh with Oh My Zsh:** Enhanced terminal experience with powerful auto-completion, syntax highlighting, and a rich plugin ecosystem for improved productivity.

* **`pre-commit` Hooks:** Automated code quality checks (linting, formatting) before every commit, ensuring consistent code style and preventing common errors.

* **`Makefile`:** Streamlined commands for common development tasks like setting up the environment, running tests, linting, formatting, and building the project.

* **Ruff:** An extremely fast Python linter and formatter, integrated into `pre-commit` for efficient code quality enforcement.

* **Hatch:** Modern Python project management tool for dependency management and packaging.

## 🚀 Getting Started

Follow these steps to set up your development environment using VS Code Dev Containers.

### Prerequisites

* [Git](https://git-scm.com/): For version control.

* [Docker Desktop](https://www.docker.com/products/docker-desktop/): Required to run Dev Containers.

* [Visual Studio Code](https://code.visualstudio.com/): Your primary IDE.

* [VS Code Dev Containers Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers): Install this extension in VS Code.

### 1. Clone the Repository

First, clone this repository to your local machine:

```zsh
git clone https://github.com/your-username/transformer-mentor.git # Replace with your actual repo URL
cd transformer-mentor
```


### 2. Open in Dev Container

1.  Open VS Code.

2.  Go to `File` > `Open Folder...` and select the `transformer-mentor` directory you just cloned.

3.  VS Code will detect the `.devcontainer` folder and prompt you to "Reopen in Container". Click this button.

    * If you don't see the prompt, open the Command Palette (`Ctrl+Shift+P` or `F1`) and select "Dev Containers: Reopen in Container".

The first time you open the project in a Dev Container, it will take some time to build the Docker image and set up the environment (installing Python dependencies, Zsh, Oh My Zsh, etc.). Subsequent opens will be much faster.

### 3. Verify Setup

Once the Dev Container is ready, open a new terminal inside VS Code (``` Ctrl+``  ``` or `View > Terminal`). You should see:

* A Zsh shell with a customized prompt (e.g., `agnoster` theme).

* Syntax highlighting as you type.

* Auto-suggestions based on your command history.

You can verify the Python environment:

```zsh
python --version
```

Expected output: `Python 3.12.x (or similar, depending on Dockerfile)`

```zsh
which python
```

Expected output: `/workspaces/transformer-mentor/.venv/bin/python`

You can also run your `Makefile` commands:

`make help`


This should display a list of available commands, confirming your `Makefile` is working.

## 📁 Project Structure

* `.devcontainer/`: Contains the `Dockerfile` and `devcontainer.json` for the VS Code Dev Container setup.

* `notebooks/`: Jupyter notebooks for explorations, tutorials, and model prototyping.

* `src/`: Python source code for reusable modules, model definitions, utilities, and training scripts.

* `tests/`: Unit and integration tests for the codebase.

* `data/`: (Ignored by Git) Placeholder for raw and processed datasets. You will download or generate data here.

* `models/`: (Ignored by Git) Placeholder for trained model weights and checkpoints.

* `.gitignore`: Specifies files and directories to be ignored by Git.

* `.pre-commit-config.yaml`: Configuration for `pre-commit` hooks to enforce code quality.

* `.zshrc-template`: Template for your Zsh shell configuration, copied into the container.

* `Makefile`: Automation scripts for common development tasks.

* `pyproject.toml`: Project metadata and dependency management using Hatch.

* `setup.sh`: A helper script executed by the Dev Container to set up Zsh, Oh My Zsh, and other tools.

## 💻 Development Workflow

1.  **Code Editing:** Write your Python code in `src/` and experiment in `notebooks/`.

2.  **Code Quality:** `pre-commit` hooks will automatically run Ruff for linting and formatting before every `git commit`. If issues are found, they will be auto-fixed (if possible) or reported, and the commit will be blocked until issues are resolved.

    * To manually run all checks: `make check`

    * To run just linting: `make lint`

    * To run just formatting: `make format`

3.  **Testing:** Write tests in `tests/` and run them with: `make test`

4.  **Jupyter Lab:** Start Jupyter Lab for interactive development: `make dev`

5.  **Clean Up:** Remove build artifacts and the virtual environment: `make clean`

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1.  Fork the repository.

2.  Create a new branch (`git checkout -b feature/your-feature-name`).

3.  Make your changes and ensure tests pass (`make test`).

4.  Commit your changes (`git commit -m "Add new feature"`). Remember `pre-commit` will help with code quality!

5.  Push to your fork (`git push origin feature/your-feature-name`).

6.  Open a Pull Request to the `main` branch of this repository.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgements

* [PyTorch](https://pytorch.org/)

* [Hugging Face](https://huggingface.co/)

* [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)

* [Oh My Zsh](https://ohmyz.sh/)

* [Ruff](https://docs.astral.sh/ruff/)

* LLMs :-)

*Happy Coding and Learning!*