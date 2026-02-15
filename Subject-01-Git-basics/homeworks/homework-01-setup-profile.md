## Homework 1: GitHub Setup & Course Repository Clone

**Due Date:** [1404-12-05]

**Objective:**  
Set up a GitHub account, install Git on your Windows computer, configure it properly, and clone the course repository. This is the foundation for all future Git‑based assignments.

**Estimated time:** 30–45 minutes

---

### Task 1: Create a GitHub Account (if you don't have one)

1. Go to [github.com](https://github.com) and click **Sign up**.
2. Follow the instructions to create a free account.
3. Verify your email address.

**Deliverable:** Your GitHub username (you’ll need it later).

---

### Task 2: Install Git on Windows

1. **Download the installer**  
   Go to [git-scm.com/download/win](https://git-scm.com/download/win). The download should start automatically.

2. **Run the installer**  
   Double‑click the downloaded file (e.g., `Git‑2.x.x‑64‑bit.exe`).

3. **Follow the setup wizard**  
   - Accept the GNU General Public License.  
   - Choose the installation path (the default is fine).  
   - **Select components:** leave the defaults checked (including “Git Bash Here” and “Git GUI Here”).  
   - **Choose the default editor:** select your preferred editor (e.g., Nano, Vim, or Notepad++). If unsure, keep the default (Vim).  
   - **Adjust your PATH environment:** select **“Git from the command line and also from 3rd‑party software”** (the second option). This allows you to use Git in Command Prompt and PowerShell as well.  
   - **Choose the SSH executable:** use the bundled OpenSSH.  
   - **Choose HTTPS transport backend:** use the native Windows Secure Channel library.  
   - **Configure line ending conversions:** select **“Checkout Windows‑style, commit Unix‑style line endings”** (recommended for Windows).  
   - **Choose the terminal emulator:** select **“Use MinTTY (the default terminal of Git Bash)”**.  
   - **Extra options:** leave the defaults (enable file system caching, enable symbolic links if desired).  
   - Click **Install**.

4. **Complete the installation**  
   Once finished, you may leave the “Launch Git Bash” box checked and click **Finish**.

5. **Verify the installation**  
   Open **Git Bash** (you can find it in the Start menu or right‑click any folder and select “Git Bash Here”).  
   In the terminal window, type:
   ```bash
   git --version
   ```
   You should see something like `git version 2.x.x.windows.1`.

**Deliverable:** Take a screenshot of the Git Bash window showing the output of `git --version`.

---

### Task 3: Configure Git with Your Identity

Set your name and email. These will be attached to every commit you make.

In Git Bash, run:
```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"
```

*Use the same email you used for GitHub.*

Verify the configuration:
```bash
git config --global --list
```

**Deliverable:** Take a screenshot of the output showing your name and email.

---

### Task 4: Clone the Course Repository

1. Open **Git Bash**.
2. Navigate to the folder where you want to store the course materials (e.g., `Documents`).  
   ```bash
   cd ~/Documents
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/smbanaie/advanced-programming.git
   ```
4. Move into the cloned folder:
   ```bash
   cd advanced-programming
   ```
5. List the contents to see what’s inside:
   ```bash
   ls -la
   ```

**Deliverable:** Take a screenshot of the Git Bash window showing the `git clone` command and the subsequent `ls` output.

---

### Submission Instructions

Create a text file or PDF containing:
- Your GitHub username.
- The three screenshots:
  1. `git --version`
  2. `git config --global --list`
  3. Successful clone and folder listing.
- Upload the file to your learning management system (or email it as instructed).

