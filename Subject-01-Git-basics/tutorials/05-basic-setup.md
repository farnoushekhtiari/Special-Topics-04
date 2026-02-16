# Tutorial 5: Git Installation and Basic Setup

## Learning Objectives
By the end of this tutorial, you will be able to:
- Install Git on your operating system
- Configure Git with your identity and preferences
- Set up SSH keys for secure GitHub authentication
- Verify your Git installation and configuration
- Troubleshoot common setup issues

## Prerequisites
- Computer with administrator privileges (for installation)
- Internet connection for downloading Git and accessing GitHub
- Text editor or IDE for configuration files

## Why Setup Matters

Proper Git setup ensures:
- **Authentication**: Secure access to GitHub repositories
- **Attribution**: Correct author information on commits
- **Workflow**: Smooth collaboration and version control experience
- **Security**: Protected communication with remote repositories

## Part 1: Installing Git

### Windows Installation

#### Option 1: Official Git for Windows (Recommended)
1. **Download the installer:**
   - Visit [https://git-scm.com/download/win](https://git-scm.com/download/win)
   - The download should start automatically
   - Alternative: Go to [git-scm.com](https://git-scm.com) → Downloads → Windows

2. **Run the installer:**
   - Double-click the downloaded `.exe` file (e.g., `Git-2.x.x-64-bit.exe`)
   - Follow the installation wizard

3. **Installation settings (recommended):**
   - **Select Components:** Keep defaults (Git Bash, Git GUI, shell integration)
   - **Default Editor:** Choose your preferred editor (VS Code, Notepad++, Vim, etc.)
   - **PATH Environment:** Select "Git from the command line and also from 3rd-party software"
   - **SSH Executable:** Use bundled OpenSSH
   - **HTTPS Transport:** Use native Windows SSL/TLS
   - **Line Ending Conversions:** "Checkout Windows-style, commit Unix-style line endings"
   - **Terminal Emulator:** "Use MinTTY (the default terminal of Git Bash)"
   - **Extra Options:** Keep defaults

4. **Complete installation:**
   - Click "Install"
   - Optionally launch Git Bash after installation

#### Option 2: Windows Package Managers

**Chocolatey:**
```powershell
choco install git
```

**Scoop:**
```powershell
scoop install git
```

### macOS Installation

#### Option 1: Official Git Installer
1. Visit [https://git-scm.com/download/mac](https://git-scm.com/download/mac)
2. Download the latest version
3. Run the installer package

#### Option 2: Homebrew (Recommended for macOS)
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Git
brew install git
```

#### Option 3: Xcode Command Line Tools
```bash
xcode-select --install
```

### Linux Installation

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install git
```

#### CentOS/RHEL/Fedora:
```bash
# CentOS/RHEL
sudo yum install git  # or dnf install git for newer versions

# Fedora
sudo dnf install git
```

#### Arch Linux:
```bash
sudo pacman -S git
```

### Verification

Open your terminal and run:
```bash
git --version
```

**Expected output:** `git version 2.x.x` (or similar)

## Part 2: Basic Git Configuration

### Setting Your Identity

Git needs to know who you are for commit attribution:

```bash
# Set your name (use your real name)
git config --global user.name "Your Full Name"

# Set your email (use the same email as your GitHub account)
git config --global user.email "your.email@example.com"
```

**Important:** Use the same email address you used to sign up for GitHub.

### Viewing Configuration

```bash
# View all global settings
git config --global --list

# View specific settings
git config --global user.name
git config --global user.email
```

### Common Configuration Options

```bash
# Set default text editor
git config --global core.editor "code --wait"  # VS Code
git config --global core.editor "vim"          # Vim
git config --global core.editor "nano"         # Nano

# Set default branch name
git config --global init.defaultBranch main

# Enable colored output
git config --global color.ui auto

# Set line ending behavior
git config --global core.autocrlf input  # Linux/Mac
git config --global core.autocrlf true   # Windows
```

## Part 3: SSH Key Setup for GitHub

SSH keys provide secure, passwordless authentication with GitHub.

### Why Use SSH?
- **Security:** More secure than passwords
- **Convenience:** No need to enter credentials for each operation
- **Automation:** Enables automated deployments and CI/CD

### Generating SSH Keys

#### Step 1: Check for Existing Keys
```bash
ls -la ~/.ssh/
```

Look for files named `id_rsa.pub`, `id_ed25519.pub`, etc.

#### Step 2: Generate New SSH Key

**Option A: RSA Key (Compatible):**
```bash
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
```

**Option B: Ed25519 Key (Modern, recommended):**
```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```

**When prompted:**
- **File location:** Press Enter (use default `~/.ssh/id_ed25519`)
- **Passphrase:** Optional but recommended for security

#### Step 3: Add Key to SSH Agent

```bash
# Start SSH agent
eval "$(ssh-agent -s)"

# Add your key
ssh-add ~/.ssh/id_ed25519  # or id_rsa if using RSA
```

#### Step 4: Copy Public Key

```bash
# Display public key
cat ~/.ssh/id_ed25519.pub

# Alternative: Copy to clipboard (Linux)
xclip -sel clip < ~/.ssh/id_ed25519.pub

# Alternative: Copy to clipboard (macOS)
pbcopy < ~/.ssh/id_ed25519.pub

# Alternative: Copy to clipboard (Windows Git Bash)
cat ~/.ssh/id_ed25519.pub | clip
```

### Adding SSH Key to GitHub

1. **Go to GitHub:**
   - Visit [https://github.com/settings/keys](https://github.com/settings/keys)
   - Click "New SSH key"

2. **Add the key:**
   - **Title:** "My Laptop" or descriptive name
   - **Key type:** Authentication Key
   - **Paste the public key** (entire content from `cat` command)
   - Click "Add SSH key"

3. **Verify the setup:**
   ```bash
   ssh -T git@github.com
   ```

   **Expected output:**
   ```
   Hi username! You've successfully authenticated, but GitHub does not provide shell access.
   ```

### Troubleshooting SSH Setup

#### "Permission denied (publickey)"
- Ensure SSH key is added to GitHub correctly
- Verify SSH agent is running and key is loaded
- Check file permissions: `chmod 600 ~/.ssh/id_ed25519`

#### "Host key verification failed"
```bash
ssh-keyscan -H github.com >> ~/.ssh/known_hosts
```

#### SSH key not found
- Ensure you're using the correct key filename
- Check if key exists: `ls -la ~/.ssh/`

## Part 4: Testing Your Setup

### Create a Test Repository

1. **Create a test directory:**
   ```bash
   mkdir git-setup-test
   cd git-setup-test
   ```

2. **Initialize repository:**
   ```bash
   git init
   ```

3. **Create a test file:**
   ```bash
   echo "# Git Setup Test" > README.md
   echo "This repository tests Git installation and configuration." >> README.md
   ```

4. **Make your first commit:**
   ```bash
   git add README.md
   git commit -m "Initial commit: Test Git setup"
   ```

### Connect to GitHub

1. **Create repository on GitHub:**
   - Go to [github.com](https://github.com) → New repository
   - Name: `git-setup-test`
   - Make it public or private
   - Don't initialize with README

2. **Connect local repository:**
   ```bash
   # Add remote (replace YOUR_USERNAME)
   git remote add origin git@github.com:YOUR_USERNAME/git-setup-test.git

   # Push to GitHub
   git push -u origin main
   ```

3. **Verify on GitHub:**
   - Check that your repository appears on GitHub
   - Verify the commit and file are there

### Cleanup Test Repository

```bash
cd ..
rm -rf git-setup-test
```

Then delete the repository from GitHub if desired.

## Part 5: Advanced Configuration

### Git Aliases

Create shortcuts for common commands:

```bash
# Short status
git config --global alias.st status

# Short log with graph
git config --global alias.lg "log --oneline --graph --decorate"

# Short diff
git config --global alias.d diff

# Amend last commit
git config --global alias.amend "commit --amend"
```

**Usage:**
```bash
git st      # instead of git status
git lg      # instead of git log --oneline --graph --decorate
```

### Global .gitignore

Create a global ignore file for files you never want to track:

```bash
# Create global gitignore
touch ~/.gitignore_global

# Add common ignore patterns
echo "# OS generated files" >> ~/.gitignore_global
echo ".DS_Store" >> ~/.gitignore_global
echo ".DS_Store?" >> ~/.gitignore_global
echo "._*" >> ~/.gitignore_global
echo ".Spotlight-V100" >> ~/.gitignore_global
echo ".Trashes" >> ~/.gitignore_global
echo "ehthumbs.db" >> ~/.gitignore_global
echo "Thumbs.db" >> ~/.gitignore_global

# Configure Git to use it
git config --global core.excludesfile ~/.gitignore_global
```

### Multiple Git Configurations

If you work with multiple accounts:

```bash
# Work configuration
git config --global includeIf."gitdir:~/work/".path ~/.gitconfig-work

# Personal configuration
git config --global includeIf."gitdir:~/personal/".path ~/.gitconfig-personal
```

## Part 6: Common Issues and Solutions

### "git: command not found"
- Git not installed or not in PATH
- Restart terminal/command prompt
- Check installation location

### "Please tell me who you are"
- User name/email not configured
- Run: `git config --global user.name "Your Name"`
- Run: `git config --global user.email "your.email@example.com"`

### "fatal: remote origin already exists"
- Remote already added, remove it first:
  ```bash
  git remote remove origin
  git remote add origin <url>
  ```

### "Permission denied"
- Wrong authentication method
- SSH key not added to GitHub
- Repository permissions incorrect

### "warning: LF will be replaced by CRLF"
- Line ending configuration issue
- Fix: `git config --global core.autocrlf input` (Linux/Mac)
- Fix: `git config --global core.autocrlf true` (Windows)

## Summary

✅ **Git Installed:** Verified installation with `git --version`
✅ **Identity Configured:** Name and email set globally
✅ **SSH Keys Generated:** Secure authentication configured
✅ **GitHub Connected:** SSH key added and tested
✅ **Repository Created:** Test repository pushed successfully

## Next Steps

Now that Git is set up, you can:
- [Learn repository basics](../tutorials/02-repository-basics.md)
- [Complete the first workshop](../workshops/workshop-01-basic-setup.md)
- Start working with real projects

## Additional Resources

- [Git Configuration Documentation](https://git-scm.com/docs/git-config)
- [SSH Key Generation Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
- [GitHub SSH Setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
- [Git Installation Guides](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

---

*Proper Git setup is crucial for a smooth development experience. Take time to configure everything correctly now to avoid issues later.*