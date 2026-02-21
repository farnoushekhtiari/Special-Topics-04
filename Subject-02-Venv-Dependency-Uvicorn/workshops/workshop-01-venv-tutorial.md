# Workshop 1: uv Tutorial Session - Python Environment Isolation

## 🎯 Objective
By the end of this workshop, students will understand what Python virtual environments are, why they're important, and how to create and manage them using uv.

## 📋 Prerequisites
- Python 3.8+ installed on your system
- Basic terminal/command prompt knowledge
- Text editor (VS Code recommended)

## 📚 Session Overview
**Duration**: 45-60 minutes
**Format**: Interactive tutorial with live demonstrations
**Focus**: Understanding uv concepts and practical usage

---

## 🚀 Modern uv: Project Initialization with uv init

### What is uv init?
uv init is uv's modern approach to project setup that combines virtual environment creation with project structure initialization.

### When to Use uv init vs uv venv

**Use `uv init` when:**
- Starting a new Python project from scratch
- You want automated project structure setup
- You need `pyproject.toml` for dependency management
- You want modern Python packaging standards

**Use `uv venv` when:**
- Working with existing projects
- You need manual control over environment setup
- Migrating from traditional virtual environments

### Step-by-Step: uv init Project Setup

#### Step 1: Initialize Your Project
```bash
# Create and navigate to your project directory
mkdir my-uv-project
cd my-uv-project

# Initialize with uv (creates pyproject.toml, .python-version, and virtual environment)
uv init
```

#### Step 2: What uv init Creates
After running `uv init`, you'll see these files:

```
my-uv-project/
├── .python-version    # Specifies Python version for the project
├── pyproject.toml     # Modern Python project configuration
├── .venv/            # Virtual environment (created when first needed)
│   ├── bin/          # Executables (Linux/Mac)
│   ├── Scripts/      # Executables (Windows)
│   ├── Lib/          # Python standard library
│   └── site-packages/# Installed packages
└── main.py           # Default entry point (optional, can be removed)
```

**Note**: The `.venv` directory is created automatically when you first run uv commands that need it (like `uv add` or `uv run`), or you can create it explicitly with `uv venv`.

#### Step 3: Understanding the Generated Files

**`.python-version`** - Specifies which Python version uv should use:
```bash
cat .python-version
# Output: 3.13
```

**`pyproject.toml`** - Modern project configuration:
```toml
[project]
name = "my-uv-project"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = []
```

**`.venv/`** - Your virtual environment (created automatically)

#### Step 4: Working with uv init Projects
```bash
# The virtual environment is automatically activated in uv commands
# No need to manually activate/deactivate for most operations

# Add dependencies
uv add requests fastapi

# Run your application
uv run python main.py

# Run scripts or commands in the environment
uv run uvicorn main:app --reload
```

#### Step 5: uv init vs Traditional uv venv

| Feature | uv init | uv venv |
|---------|---------|---------|
| Project structure | ✅ Creates `pyproject.toml` | ❌ Manual setup needed |
| Virtual environment | ✅ Auto-created | ✅ Manual creation |
| Python version | ✅ Managed via `.python-version` | ❌ Manual specification |
| Dependencies | ✅ Modern `uv add` workflow | ⚠️ Traditional pip |
| Activation | ✅ Auto-handled by uv | ✅ Manual activation needed |

### Real-World uv init Example
```bash
# Create a FastAPI project
mkdir fastapi-blog
cd fastapi-blog
uv init

# Add dependencies
uv add fastapi uvicorn

# Create your app (replace the default main.py)
# Write your FastAPI code...

# Run the application
uv run uvicorn main:app --reload
```

---

## 🔍 What is uv? Why Do We Need It?

### The Problem: Global Package Conflicts
```python
# Without virtual environments, all packages go to system Python
# This can cause conflicts between different projects

# Project A needs requests==2.25.0
# Project B needs requests==2.28.0
# System can only have one version installed!
```

### The Solution: Isolated Environments
```
System Python (Global)
├── Project A Environment
│   ├── Python 3.8
│   ├── requests==2.25.0
│   └── other dependencies
│
├── Project B Environment
│   ├── Python 3.8
│   ├── requests==2.28.0
│   └── other dependencies
```

### Key Benefits of uv
1. **Isolation**: Each project has its own dependencies
2. **Reproducibility**: Same environment across different machines
3. **No Conflicts**: Different projects can use different package versions
4. **Clean Uninstall**: Remove project = remove its environment
5. **Development vs Production**: Different environments for different purposes

---

## 🛠️ Step-by-Step: Creating Your First Virtual Environment

### Step 1: Open Terminal/Command Prompt
**Windows**: Press `Win + R`, type `cmd`, press Enter
**macOS**: Press `Cmd + Space`, type `terminal`, press Enter
**Linux**: Press `Ctrl + Alt + T`

### Step 2: Navigate to Your Project Directory
```bash
# Create a new directory for our project
mkdir my-first-project
cd my-first-project
```

### Step 3: Create Virtual Environment
```bash
# Create a virtual environment named 'uv'
uv venv

# This creates a 'uv' folder in your project directory
```

**What happens when you run this command?**
- Python creates a new folder called `uv`
- Inside this folder: complete Python installation
- Isolated from system Python packages

### Step 4: Activate the Virtual Environment

#### Windows (Command Prompt):
```cmd
.venv\Scripts\activate
```

#### Windows (PowerShell):
```powershell
.venv\Scripts\activate.ps1
```

#### macOS/Linux:
```bash
source .venv/bin/activate
```

**What changes when activated?**
- Command prompt shows `(uv)` at the beginning
- `python` and `pip` now point to virtual environment
- Packages installed will go to this environment

### Step 5: Verify Activation
```bash
# Check which Python we're using
which python  # Linux/Mac: Shows path to uv python
where python  # Windows: May show nothing if Python not in PATH
python --version

# Alternative verification (works on all platforms):
# Check if you're in the virtual environment
echo $VIRTUAL_ENV  # Linux/Mac: Shows .venv path if activated
echo %VIRTUAL_ENV% # Windows CMD: Shows .venv path if activated

# Check pip location
which pip     # Linux/Mac: Shows pip location
where pip     # Windows: May show nothing if pip not in PATH
pip --version
```

**Note for Windows users**: If `where python` shows nothing, this is normal! It means Python isn't in your system PATH. The virtual environment is still working - you can verify with `python --version` and `echo %VIRTUAL_ENV%`.

### Step 6: Install Packages
```bash
# Install a package
uv add requests

# Check installed packages
pip list

# Check if package is in our environment
pip show requests
```

### Step 7: Deactivate Environment
```bash
# Deactivate the virtual environment
deactivate

# Notice: (uv) disappears from command prompt
```

---

## 🔍 Understanding the uv Folder Structure

After creating uv, explore the folder structure:

```
uv/
├── bin/              # macOS/Linux executables
│   ├── python        # Python executable
│   ├── pip           # pip executable
│   └── activate      # Activation script
├── Scripts/          # Windows executables
│   ├── python.exe    # Python executable
│   ├── pip.exe       # pip executable
│   └── activate.bat  # Activation script
├── Lib/              # Python standard library
├── site-packages/    # Installed packages go here
└── pyuv.cfg        # Configuration file
```

---

## 📝 Common uv Commands Cheat Sheet

```bash
# Create virtual environment
uv venv

# Activate (Windows CMD)
.venv\Scripts\activate

# Activate (Windows PowerShell)
.venv\Scripts\activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Deactivate
deactivate

# Check Python path
which python   # Linux/Mac
where python   # Windows (may show nothing if Python not in PATH)

# Check pip path
which pip      # Linux/Mac
where pip      # Windows (may show nothing if pip not in PATH)

# Alternative verification (all platforms)
echo $VIRTUAL_ENV   # Linux/Mac: Shows .venv path if activated
echo %VIRTUAL_ENV%  # Windows: Shows .venv path if activated

# List installed packages
pip list

# Install package
uv add package-name

# Uninstall package
pip uninstall package-name
```

---

## 🚨 Common Issues and Solutions

### Issue 1: Permission Denied (Windows)
**Problem**: `.venv\Scripts\activate` gives permission error
**Solution**: Run Command Prompt as Administrator, or use PowerShell with execution policy

### Issue 2: Command Not Found
**Problem**: `uv venv` says command not found
**Solution**: Make sure Python is installed and in PATH

### Issue 3: Wrong Python Version
**Problem**: Virtual environment uses wrong Python version
**Solution**: Specify Python version: `uv venv`

### Issue 4: Packages Still Install Globally
**Problem**: `uv add` installs to system Python
**Solution**: Make sure environment is activated (check `(uv)` in prompt)

### Issue 5: where python Shows Nothing (Windows)
**Problem**: `where python` doesn't show any output on Windows
**Solution**: This is normal! On Windows, Python often isn't in system PATH by default. Verify activation with:
- `python --version` (should show Python version)
- `echo %VIRTUAL_ENV%` (should show `.venv` path)
- `pip --version` (should show virtual environment pip)

---

## 🎯 Best Practices

### 1. One uv Per Project
```
project-a/
├── .venv/
├── src/
└── requirements.txt

project-b/
├── .venv/
├── src/
└── requirements.txt
```

### 2. Never Commit uv Folder
Add `.venv/` to `.gitignore`

### 3. Use Descriptive Names When Needed
```bash
# For different environments
uv venv-dev
uv venv-prod
```

### 4. Always Activate Before Working
```bash
# Habit: cd to project, then activate
cd myproject
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

---

## 🔍 Real-World Scenarios

### Scenario 1: Django Project
```bash
mkdir django-blog
cd django-blog
uv venv
source .venv/bin/activate  # Linux/Mac
uv add django
django-admin startproject blog .
python manage.py runserver
```

### Scenario 2: FastAPI Project
```bash
mkdir fastapi-app
cd fastapi-app
uv venv
source .venv/bin/activate  # Linux/Mac
uv add fastapi uvicorn
# Create main.py with FastAPI app
uvicorn main:app --reload
```

### Scenario 3: Data Science Project
```bash
mkdir data-analysis
cd data-analysis
uv venv
source .venv/bin/activate  # Linux/Mac
uv add pandas numpy matplotlib jupyter
jupyter notebook
```

---

## 📋 Verification Checklist

After completing this workshop, you should be able to:

- [ ] Create a new virtual environment with `uv venv`
- [ ] Activate the environment (platform-specific command)
- [ ] Verify activation by checking `(uv)` in command prompt
- [ ] Install packages with `uv add` while activated
- [ ] Confirm packages are installed in the virtual environment
- [ ] Deactivate the environment with `deactivate`
- [ ] Explain why virtual environments are important

---

## 🎓 Key Takeaways

1. **uv creates isolated Python environments** for each project
2. **Activation changes which Python/pip you're using**
3. **Packages installed in uv stay in that environment**
4. **Deactivation returns to system Python**
5. **One uv per project** prevents dependency conflicts

---

## 🚀 Next Steps

Now that you understand uv, in the next session we'll explore:
- What dependency management really means
- Why modern tools like UV are better than traditional pip
- How to use UV for faster, more reliable package management

## 📚 Additional Resources

- [Python uv Documentation](https://docs.python.org/3/library/uv.html)
- [Real Python: Virtual Environments](https://realpython.com/python-virtual-environments-a-primer/)
- [Python Packaging Guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
