## Homework 3: Personal Website with GitHub Pages

**Due Date:** [Insert due date – typically 1 week after Homework 2]

**Objective:**  
Create a personal website using GitHub Pages. You’ll learn to set up a repository, use a provided template, customise it, and publish it live on the web.

**Estimated time:** 45–60 minutes

---

### Part 1: Understanding GitHub Pages

GitHub Pages is a free hosting service that turns your repository into a website. If you create a repository named `username.github.io` (where `username` is your GitHub username), GitHub will automatically serve the files in that repository as a website at `https://username.github.io`.

---

### Task 1: Create the Special Repository

1. Go to [github.com/new](https://github.com/new).
2. Repository name: **`username.github.io`** (replace `username` with your actual GitHub username, all lowercase).
   *Example: if your username is `johndoe`, the repo name is `johndoe.github.io`.*
3. Description: “My personal website”.
4. Visibility: **Public** (GitHub Pages requires public repositories unless you have a paid plan).
5. **Do NOT** initialise with a README (we’ll add files manually).
6. Click **Create repository**.

---

### Task 2: Get the Template Files

I’ve prepared a simple but elegant template for you.  
[Download the template ZIP](https://example.com/template.zip) or create the files manually as shown below.

**Template structure:**
```
personal-site/
├── index.html          # main page
├── style.css           # styling
├── script.js           # (optional) interactivity
└── README.md           # instructions (you can delete later)
```

#### `index.html`
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Name | Personal Site</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>Hi, I'm <span id="name">Your Name</span></h1>
        <p>Student | Developer | Lifelong Learner</p>
    </header>
    <main>
        <section id="about">
            <h2>About Me</h2>
            <p>Write a short paragraph about yourself here. What do you study? What are you passionate about?</p>
        </section>
        <section id="projects">
            <h2>Projects</h2>
            <ul>
                <li><a href="#">Project 1</a> – brief description</li>
                <li><a href="#">Project 2</a> – brief description</li>
                <li><a href="#">Project 3</a> – brief description</li>
            </ul>
        </section>
        <section id="contact">
            <h2>Contact</h2>
            <p>Find me on <a href="https://github.com/yourusername">GitHub</a> or send an email to <a href="mailto:you@example.com">you@example.com</a>.</p>
        </section>
    </main>
    <footer>
        <p>&copy; 2025 Your Name</p>
    </footer>
    <script src="script.js"></script>
</body>
</html>
```

#### `style.css`
```css
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    line-height: 1.6;
    color: #333;
    background-color: #f4f4f4;
}
header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    padding: 3rem 1rem;
}
header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}
header p {
    font-size: 1.2rem;
    opacity: 0.9;
}
main {
    max-width: 800px;
    margin: 2rem auto;
    padding: 0 1rem;
}
section {
    background: white;
    margin-bottom: 2rem;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
h2 {
    color: #667eea;
    margin-top: 0;
}
a {
    color: #764ba2;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
footer {
    text-align: center;
    padding: 1rem;
    background: #333;
    color: white;
}
```

#### `script.js` (optional)
```js
// Simple interactive example: update greeting based on time
document.addEventListener('DOMContentLoaded', function() {
    const nameSpan = document.getElementById('name');
    const hour = new Date().getHours();
    let greeting = '';

    if (hour < 12) greeting = 'Good morning';
    else if (hour < 18) greeting = 'Good afternoon';
    else greeting = 'Good evening';

    nameSpan.textContent = `${greeting}, I'm Your Name`;
});
```

**Instructions:**  
Create these three files on your computer inside a folder named `personal-site`. You can use any text editor (Notepad, VS Code, etc.).

---

### Task 3: Clone Your New Repository

In your terminal, go to the folder where you want to keep the repository (e.g., `~/Documents`). Then:

```bash
git clone https://github.com/username/username.github.io.git
cd username.github.io
```

Replace `username` with your GitHub username.

---

### Task 4: Add Your Template Files

Copy the three files (`index.html`, `style.css`, `script.js`) from the `personal-site` folder into the cloned repository folder.

You can use your file explorer or terminal commands:

```bash
# Example: if your template is in ~/Desktop/personal-site
cp ~/Desktop/personal-site/* .   # Linux/macOS
copy C:\Users\You\Desktop\personal-site\* .   # Windows (Command Prompt)
```

---

### Task 5: Customise the Website

Open the files in a text editor and replace placeholder text with your own information:

- **index.html**:
  - Change “Your Name” in the `<h1>` to your real name.
  - Update the tagline under the header.
  - Write your own “About Me” paragraph.
  - Replace the project links and descriptions with real ones (or remove the section).
  - Update the email and GitHub links.
- **style.css**: You can tweak colours, fonts, etc. if you like.
- **script.js**: Change the name in the greeting or remove the script entirely.

**Tip:** Keep it simple and professional. This will be your public face!

---

### Task 6: Commit and Push Your Changes

```bash
git add .
git commit -m "Add personal website template"
git push origin main
```

If your default branch is `master`, use `git push origin master` instead.

---

### Task 7: Enable GitHub Pages

1. Go to your repository on GitHub.
2. Click **Settings** (the tab near the top right).
3. In the left sidebar, click **Pages**.
4. Under **Branch**, select `main` (or `master`) and `/ (root)`, then click **Save**.
5. Wait a minute or two, then refresh the page. You’ll see a message like:
   > Your site is published at `https://username.github.io/`

Visit that URL – you should see your live personal website!

**Deliverable:** Take a screenshot of your live website in a browser.

---

### Optional Enhancements (for extra practice)

- Add more pages (e.g., `projects.html`, `contact.html`) and link them.
- Use a custom domain (if you have one).
- Add a profile picture.
- Explore Jekyll themes for GitHub Pages.

---

### Submission Instructions

Submit a document containing:
- The URL of your live GitHub Pages site (e.g., `https://username.github.io`).
- A screenshot of the site.
- The URL of your repository (e.g., `https://github.com/username/username.github.io`).

