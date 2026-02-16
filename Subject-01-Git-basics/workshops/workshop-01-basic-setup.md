# Workshop 1: Git Installation & Basic Setup

## Duration: 30-45 minutes

## Objective
By the end of this workshop, you will have Git installed and configured on your system, ready for development work. You'll also learn about authentication options for working with GitHub repositories.

## Prerequisites
- Computer with internet access
- Administrator privileges (for installation)
- Text editor (optional, for configuration)
- GitHub account (for authentication setup)

## Materials Needed
- Computer (Windows, macOS, or Linux)
- Internet connection
- Command line terminal (built-in on all platforms)

---

## Part 1: Installing Git

### Step 1: Identify Your Operating System
**What you'll do:** Determine which installation guide to follow.

**Instructions:**
1. Check your operating system:
   - Windows: Press `Win + R`, type `winver`, press Enter
   - macOS: Click Apple menu → About This Mac
   - Linux: Open terminal and run `cat /etc/os-release`

2. Note your OS version for reference during troubleshooting

**Expected Result:**
You know your OS and version (e.g., "Windows 11", "macOS Ventura", "Ubuntu 22.04")

---

### Step 2: Download and Install Git

**Windows Users:**

1. Open your web browser and go to: https://git-scm.com/download/win
2. Click the download link (should start automatically)
3. Locate the downloaded file (usually in Downloads folder)
4. Double-click the `.exe` file to start installation
5. Follow the installation wizard:
   - Click "Next" on the information screen
   - Keep all components selected, click "Next"
   - Choose your preferred editor, click "Next"
   - **Important:** Select "Git from the command line and also from 3rd-party software", click "Next"
   - Keep other defaults, click "Install"
   - Click "Finish"

**macOS Users:**

**Option A: Homebrew (Recommended)**
1. Open Terminal (Applications → Utilities → Terminal)
2. Install Homebrew (if not installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Install Git:
   ```bash
   brew install git
   ```

**Option B: Xcode Command Line Tools**
1. Open Terminal
2. Run: `xcode-select --install`
3. Click "Install" when prompted
4. Wait for installation to complete

**Linux Users:**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install git
```

**CentOS/RHEL/Fedora:**
```bash
sudo dnf install git  # or sudo yum install git
```

**Arch Linux:**
```bash
sudo pacman -S git
```

---

### Step 3: Verify Installation

**What you'll do:** Confirm Git is installed correctly.

**Instructions:**
1. Open your terminal/command prompt:
   - Windows: Search for "Command Prompt" or "PowerShell"
   - macOS: Applications → Utilities → Terminal
   - Linux: Search for "Terminal" in applications

2. Type the following command and press Enter:
   ```bash
   git --version
   ```

**Expected Output:**
```
git version 2.x.x      # Version number may vary
```

**If you see an error:**
- Windows: Try running Command Prompt as Administrator
- macOS/Linux: Check if you used `sudo` for installation
- All platforms: Restart your terminal and try again

---

## Part 2: Basic Git Configuration

### Step 4: Configure User Identity

**What you'll do:** Set up your name and email for Git commits.

**Instructions:**
1. In your terminal, run these commands (replace with your actual information):

   ```bash
   git config --global user.name "Your Full Name"
   git config --global user.email "your.email@example.com"
   ```

   **Example:**
   ```bash
   git config --global user.name "John Doe"
   git config --global user.email "john.doe@university.edu"
   ```

2. Verify your configuration:
   ```bash
   git config --global user.name
   git config --global user.email
   ```

**Expected Output:**
```
John Doe
john.doe@university.edu
```

### Step 5: Set Default Branch Name

**Instructions:**
```bash
git config --global init.defaultBranch main
```

**Verification:**
```bash
git config --global init.defaultBranch
```

**Expected Output:**
```
main
```

### Step 6: Configure Line Endings (Platform-Specific)

**Windows:**
```bash
git config --global core.autocrlf true
```

**macOS/Linux:**
```bash
git config --global core.autocrlf input
```

### Step 7: Enable Helpful Features

**Instructions:**
```bash
git config --global pull.rebase false
git config --global color.ui auto
git config --global core.editor "nano"  # or "code" for VS Code
```

---

### Step 8: View Complete Configuration

**Instructions:**
```bash
git config --global --list
```

**Expected Output:** (similar to)
```
user.name=John Doe
user.email=john.doe@university.edu
init.defaultBranch=main
core.autocrlf=true
pull.rebase=false
color.ui=auto
core.editor=nano
```

---

## Part 3: Testing Your Setup

### Step 9: Create a Test Repository

**What you'll do:** Create your first Git repository to test everything works.

**Instructions:**
1. Create a new directory for testing:
   ```bash
   mkdir git-test
   cd git-test
   ```

2. Initialize Git repository:
   ```bash
   git init
   ```

3. Create a test file:
   ```bash
   echo "# My First Git Repository" > README.md
   echo "print('Hello, Git!')" > hello.py
   ```

4. Check status:
   ```bash
   git status
   ```

**Expected Output:**
```
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        README.md
        hello.py

nothing added to commit but untracked files present (use "git add" to track)
```

### Step 10: Make Your First Commit

**Instructions:**
1. Add files to staging:
   ```bash
   git add .
   ```

2. Check status again:
   ```bash
   git status
   ```

3. Commit the files:
   ```bash
   git commit -m "Initial commit: Add README and hello.py"
   ```

4. Check the log:
   ```bash
   git log --oneline
   ```

**Expected Output:**
```
abc1234 Initial commit: Add README and hello.py
```

### Step 11: Clean Up Test Repository

**Instructions:**
```bash
cd ..
rm -rf git-test
```

---

## Part 4: Optional Advanced Setup

### Setting Up SSH Keys (Recommended for GitHub)

**Why SSH?** More secure than passwords, no need to enter credentials repeatedly.

**Instructions:**

1. Generate SSH key:
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```
   Press Enter for all prompts (or set a passphrase)

2. Copy public key to clipboard:
   - macOS: `pbcopy < ~/.ssh/id_ed25519.pub`
   - Linux: `xclip -sel clip < ~/.ssh/id_ed25519.pub`
   - Windows: `type %USERPROFILE%\.ssh\id_ed25519.pub`

3. Go to GitHub.com → Settings → SSH and GPG keys → New SSH key
4. Paste your public key and save

4. Test connection:
   ```bash
   ssh -T git@github.com
   ```

**Expected Output:**
```
Hi username! You've successfully authenticated, but GitHub does not provide shell access.
```

### Setting Up Personal Access Tokens (Alternative to SSH)

**Why PAT?** GitHub requires token-based authentication for HTTPS operations. PATs are essential for cloning repositories, pushing commits, and other GitHub interactions.

**When to use PAT vs SSH:**
- **Use SSH** for personal development (more secure, convenient)
- **Use PAT** when SSH isn't available or for CI/CD systems
- **Both work** - you can set up both methods

**Note:** GitHub removed password authentication in 2021. You must use either SSH keys or Personal Access Tokens for Git operations.

**For detailed PAT setup:** See [Tutorial 7: GitHub Personal Access Tokens](../tutorials/07-github-personal-access-tokens.md)

**Quick PAT reminder:**
1. Go to GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token with `repo` scope
3. Use in URLs: `https://YOUR_USERNAME:YOUR_PAT@github.com/YOUR_USERNAME/repo.git`

---

## Verification Checklist

- [ ] Git is installed (`git --version` works)
- [ ] User name is configured
- [ ] User email is configured
- [ ] Default branch is set to "main"
- [ ] Line endings are configured for your platform
- [ ] You can create a repository (`git init`)
- [ ] You can add files (`git add`)
- [ ] You can commit files (`git commit`)
- [ ] You can view history (`git log`)
- [ ] Authentication method configured (SSH or PAT)
- [ ] Can authenticate with GitHub (optional)

---

## Troubleshooting

### "git is not recognized as an internal or external command" (Windows)
**Solution:**
1. Check if Git is installed: Control Panel → Programs → Programs and Features
2. If installed, add to PATH:
   - Open System Properties → Advanced → Environment Variables
   - Find "Path" in System variables
   - Add: `C:\Program Files\Git\cmd`
   - Restart Command Prompt

### Permission denied during installation
**Solution:** Run installer as Administrator (Windows) or use `sudo` (macOS/Linux)

### "xcode-select: command not found" (macOS)
**Solution:** Install Xcode from App Store first, or use Homebrew method

### SSH connection fails
**Solution:**
- Check if SSH key exists: `ls ~/.ssh/`
- Verify public key is added to GitHub
- Try: `ssh-add ~/.ssh/id_ed25519` (if using passphrase)

### "Support for password authentication was removed" error
**Solution:**
- GitHub no longer accepts passwords for Git operations
- Set up SSH keys (see Part 4 above) OR
- Generate a Personal Access Token (see Tutorial 7)
- Use PAT in repository URLs: `https://username:token@github.com/username/repo.git`

### Configuration not working
**Solution:**
- Check for typos in commands
- Use `git config --global --list` to verify
- Try without `--global` for repository-specific config

---

## What You've Accomplished

✅ **Git Installation:** Installed Git on your system
✅ **Basic Configuration:** Set up user identity and preferences
✅ **First Repository:** Created and committed to your first Git repository
✅ **Command Line Skills:** Used terminal commands successfully
✅ **Authentication Options:** Learned about SSH keys and Personal Access Tokens
✅ **Optional SSH Setup:** Configured secure connection to GitHub (if completed)

---

## Next Steps

Now that Git is set up, you can:

1. [Set up Personal Access Tokens](../tutorials/07-github-personal-access-tokens.md) for GitHub authentication
2. [Create your first repository](../workshops/workshop-02-first-repo.md)
3. [Learn about working with files](../tutorials/03-working-with-files.md)
4. Practice more with the [basic concepts](../tutorials/01-git-intro.md)

---

## Help & Support

- **Installation Issues:** Check the [platform-specific guides](../installation/)
- **Command Problems:** Review the troubleshooting section above
- **GitHub Setup:** Visit https://docs.github.com/en/get-started
- **Course Forum:** Ask questions in the course discussion forum

**Remember:** Every expert was once a beginner. Take your time, and don't hesitate to ask for help!

---

## Workshop Complete! 🎉

You've successfully set up Git on your system. This is a major milestone in your journey to becoming proficient with version control and collaborative development.
