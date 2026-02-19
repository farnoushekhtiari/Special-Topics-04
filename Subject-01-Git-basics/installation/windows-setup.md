# Git Installation Guide for Windows

## Overview
This guide will help you install Git on Windows and set up your development environment for the Git Basics course.

## Method 1: Official Git for Windows Installer (Recommended)

### Step 1: Download Git
1. Open your web browser and go to: https://git-scm.com/download/win
2. The download should start automatically
3. Alternatively, you can visit: https://github.com/git-for-windows/git/releases/latest
4. Download the latest `.exe` file

### Step 2: Run the Installer
1. Locate the downloaded file (usually in your Downloads folder)
2. Double-click the `.exe` file to start the installation
3. If prompted by User Account Control, click "Yes"

### Step 3: Installation Wizard
Follow these steps in the installation wizard:

#### Information Screen
- Click "Next"

#### Select Components
- Keep all default options checked
- Click "Next"

#### Default Editor
- Choose your preferred text editor (Notepad++ or VS Code recommended)
- Click "Next"

#### PATH Environment
- **IMPORTANT**: Select "Git from the command line and also from 3rd-party software"
- This ensures Git commands work in Command Prompt, PowerShell, and other terminals
- Click "Next"

#### HTTPS Transport Backend
- Keep "Use the OpenSSL library" selected
- Click "Next"

#### Line Ending Conversions
- Select "Checkout Windows-style, commit Unix-style line endings"
- Click "Next"

#### Terminal Emulator
- Select "Use Windows' default console window"
- Click "Next"

#### Default Pull Behavior
- Select "Default (fast-forward or merge)"
- Click "Next"

#### Credential Helper
- Select "Git Credential Manager"
- Click "Next"

#### Extra Options
- Keep all default options
- Click "Next"

#### Experimental Options
- Keep all unchecked (default)
- Click "Install"

### Step 4: Complete Installation
- Wait for the installation to complete
- Click "Finish"

## Method 2: Using Winget (Windows Package Manager)

If you have Windows 10/11 with the latest updates, you can use the Windows Package Manager:

1. Open PowerShell or Command Prompt as Administrator
2. Run: `winget install --id Git.Git -e --source winget`
3. Wait for installation to complete

## Method 3: Using Chocolatey

If you have Chocolatey installed:

1. Open Command Prompt or PowerShell as Administrator
2. Run: `choco install git`
3. Wait for installation to complete

## Verification: Test Your Git Installation

### Step 1: Open Command Prompt or PowerShell
- Press `Win + R`, type `cmd`, and press Enter
- Or search for "Command Prompt" in the Start menu

### Step 2: Check Git Version
```bash
git --version
```

**Expected Output:**
```
git version 2.x.x.windows.x
```

### Step 3: Check Git Configuration
```bash
git config --list --show-origin
```

This should show some default configurations.

## Basic Git Configuration

### Step 1: Set Your Identity
Open Command Prompt and run these commands (replace with your actual information):

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"
```

**Example:**
```bash
git config --global user.name "John Doe"
git config --global user.email "john.doe@email.com"
```

### Step 2: Set Default Branch Name
```bash
git config --global init.defaultBranch main
```

### Step 3: Enable Helpful Features
```bash
git config --global core.autocrlf true
git config --global pull.rebase false
```

### Step 4: Verify Configuration
```bash
git config --global --list
```

## Setting Up SSH Keys (Optional but Recommended)

### Why Use SSH?
- More secure than HTTPS
- No need to enter credentials for each push/pull
- Required for some Git hosting services

### Generate SSH Key
1. Open Git Bash (installed with Git)
2. Run: `ssh-keygen -t ed25519 -C "your.email@example.com"`
3. Press Enter for default location
4. Enter a passphrase (optional but recommended)
5. Copy the public key: `cat ~/.ssh/id_ed25519.pub`

### Add to GitHub/GitLab
1. Go to GitHub.com → Settings → SSH and GPG keys
2. Click "New SSH key"
3. Paste your public key
4. Save

## Troubleshooting

### Git Command Not Recognized
If you get "'git' is not recognized as an internal or external command":

1. Check if Git is installed: Go to Control Panel → Programs → Programs and Features
2. Look for "Git" in the list
3. If missing, reinstall using the steps above
4. Make sure you selected "Git from the command line and also from 3rd-party software" during installation

### Permission Denied Errors
- Run Command Prompt as Administrator
- Or use Git Bash instead of Command Prompt

### PATH Issues
If Git commands work in Git Bash but not in Command Prompt:
1. Open System Properties → Advanced → Environment Variables
2. Find "Path" in System variables
3. Add: `C:\Program Files\Git\cmd` (or your Git installation path)
4. Restart Command Prompt

## Next Steps

Now that Git is installed and configured, you can:

1. [Create your first repository](../workshops/workshop-02-first-repo.md)
2. [Learn basic Git concepts](../tutorials/01-git-intro.md)
3. [Set up GitHub account](https://github.com) and connect it to Git

## Additional Resources

- [Official Git Documentation](https://git-scm.com/doc)
- [GitHub Git Cheat Sheet](https://github.github.com/training-kit/downloads/github-git-cheat-sheet/)
- [Interactive Git Tutorial](https://learngitbranching.js.org/)

