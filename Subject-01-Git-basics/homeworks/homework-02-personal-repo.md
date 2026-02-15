## Homework 2: Personal Project Repository & Copilot‑Enhanced README

**Due Date:** [1404-12-05]

**Objective:**  
Take an existing personal project (or create a simple one), host it on GitHub, and use GitHub Copilot to write a professional README. This simulates real‑world project sharing.

**Estimated time:** 60–90 minutes

---

### Task 1: Prepare Your Project

Choose one of the following:
- A small project you wrote for a previous course (e.g., a calculator, a to‑do list, a data analysis script).
- If you don’t have any, create a very simple “Hello World” program in any language (Python, JavaScript, etc.) with at least two files.

Make sure the project is in a folder on your computer.  
Example project structure:
```
my-project/
├── main.py          # or index.html, app.js, etc.
├── utils.py         # optional extra file
└── README.md        # (we will replace this later)
```

---

### Task 2: Create a New Repository on GitHub

1. Log in to GitHub.
2. Click the **+** icon in the top‑right corner and select **New repository**.
3. Repository name: Choose something like `my‑first‑project` or use your project’s name.
4. Description: Add a short description (e.g., “My first project for Advanced Programming”).
5. Visibility: **Public**.
6. **Do NOT** check “Add a README”, “.gitignore”, or “license” – we want an empty repository.
7. Click **Create repository**.

You’ll see a page with instructions. Keep this page open – you’ll need the repository URL.

---

### Task 3: Clone the Repository Locally

In your terminal, navigate to the folder where you want to store the repo (e.g., `~/Documents`), then run:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name.

Now move into that folder:
```bash
cd YOUR_REPO_NAME
```

---

### Task 4: Add Your Project Files

Copy your project files into this new folder. You can use the file explorer or terminal commands like `cp` (Linux/macOS) or `copy` (Windows).

For example, if your project is in `~/Desktop/my-project`:
```bash
cp ~/Desktop/my-project/* .   # Linux/macOS
copy C:\Users\You\Desktop\my-project\* .   # Windows (in Command Prompt, not Git Bash)
```

Check that files are copied:
```bash
ls -la   # or dir on Windows
```

---

### Task 5: Commit and Push

Now we’ll save the files to Git and upload them to GitHub.

```bash
# Tell Git to track all files
git add .

# Commit with a message
git commit -m "Add my personal project files"

# Push to GitHub
git push origin main
```

If you’re using an older Git version, the default branch might be called `master`. Use `git branch` to see your branch name.

Go back to your repository page on GitHub and refresh – you should see your files!

**Deliverable:** Take a screenshot of your GitHub repository showing the uploaded files.

---

### Task 6: Create a Great README Using GitHub Copilot

GitHub Copilot is an AI pair programmer. We’ll use it inside the GitHub web editor to write a professional README.

1. On your repository page, click the **README.md** file (if you didn’t have one, click **Add file** → **Create new file** and name it `README.md`).
2. Click the pencil icon (**Edit this file**) to open the web editor.
3. You should see a little Copilot icon (✨) in the editor. If not, make sure you have [Copilot enabled](https://github.com/features/copilot) (free for students via GitHub Student Developer Pack).
4. Start typing a description of your project, for example:
   ```markdown
   # My First Project
   This project is a simple calculator that...
   ```
5. As you write, Copilot will suggest completions. Press `Tab` to accept a suggestion.
6. Ask Copilot to generate sections:
   - Type `## Features` and let Copilot list possible features.
   - Type `## Installation` and let it write installation steps.
   - Type `## Usage` and let it describe how to run the program.
   - Type `## License` and let it suggest a license (e.g., MIT).
7. Feel free to tweak the suggestions to match your actual project.

**Goal:** Your README should look professional and include at least:
- Project title and description
- Features
- How to install/run
- How to use
- Credits (you)
- License

When you’re happy, scroll down, add a commit message like “Add README with Copilot”, and click **Commit changes**.

**Deliverable:** Take a screenshot of your final README (or the repository page showing it).

---

### Submission Instructions

Submit a document containing:
- The URL of your GitHub repository.
- Screenshot of your repository with uploaded files (from Task 5).
- Screenshot of your README (from Task 6).
- (Optional) A brief reflection: How did Copilot help? What did you learn?

---
