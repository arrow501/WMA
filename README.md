# WMA Project

This repository contains various computer vision projects and exercises.

Not all code is my own, some has been given to me by the teacher to do with as I please.

## Setup Instructions

### Setting up the virtual environment with UV

UV is a fast Python package installer and resolver that can be used as an alternative to pip and virtualenv.

#### 1. Install UV (if not already installed)

```bash
# Install UV using the official installer (macOS/Linux)
curl -fsSL https://raw.githubusercontent.com/astral-sh/uv/main/install.sh | bash
# or (Cross-platform)
pip install uv
```

#### 2. Create and activate a virtual environment

```bash
# Create a new virtual environment in the .venv directory
uv venv
```

Activate the virtual environment based on your shell:

**Fish shell:**
```fish
source .venv/bin/activate.fish
```

**Bash/Zsh:**
```bash
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.venv\Scripts\activate.bat
```

#### 3. Install dependencies

```bash
# Install dependencies from requirements.txt
uv pip install -r requirements.txt
```

## Running the Applications

After setting up the environment, you can run the various applications in the repository.

For example:
```bash
python CW04-ProjektAimbot/aimbot.py
```

or

```bash
python CW05-CoinCounter/coin_counter.py
```
