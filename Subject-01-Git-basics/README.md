# Subject 1: Git Basics & Project Workflows

## Overview

This subject introduces version control fundamentals using Git, focusing on essential skills for collaborative software development. Students will learn to manage code changes, work with branches, resolve conflicts, and implement team workflows using Git and GitHub.

## Learning Objectives

By the end of this subject, you will be able to:

- **Understand Git Fundamentals**: Explain version control concepts, repository structure, and Git's role in software development
- **Manage Code Changes**: Create commits, track file states, and navigate Git history
- **Work with Branches**: Create, switch, merge, and rebase branches for feature development
- **Resolve Conflicts**: Handle merge conflicts effectively and safely
- **Collaborate Effectively**: Use pull requests, team workflows, and best practices for collaborative development
- **Configure Git**: Set up Git properly for personal and team use

## Prerequisites

- Basic familiarity with command-line interfaces
- Text editor installed (VS Code, Notepad++, etc.)
- Internet connection for GitHub access
- No prior Git knowledge required

## Subject Structure

### 📚 Tutorials (Conceptual Learning)

1. **[Git Introduction](tutorials/01-git-intro.md)**
   - What is Git and version control?
   - Key concepts: repositories, commits, branches
   - Git vs other version control systems

2. **[Repository Basics](tutorials/02-repository-basics.md)**
   - Creating and cloning repositories
   - Understanding the `.git` directory
   - Local vs remote repositories

3. **[Working with Files](tutorials/03-working-with-files.md)**
   - Staging and committing changes
   - File status and lifecycle
   - Writing meaningful commit messages

4. **[Git States & Workflow](tutorials/04-git-states.md)**
   - Working directory, staging area, and repository
   - Git status, diff, and log commands
   - Undoing changes and commits

5. **[Basic Setup](tutorials/05-basic-setup.md)**
   - Installing Git on different platforms
   - Configuring identity and preferences
   - Setting up SSH keys for GitHub
   - Verifying installation and troubleshooting

6. **[Branching Fundamentals](tutorials/06-branching-fundamentals.md)**
   - Understanding branches and their purpose
   - Creating, switching, and managing branches
   - Branch visualization and history
   - Branching strategies and best practices

7. **[GitHub Personal Access Tokens](tutorials/07-github-personal-access-tokens.md)**
   - Understanding PAT authentication requirements
   - Generating and managing personal access tokens
   - Using tokens for Git operations and cloning
   - Security best practices and token management

### 🛠️ Workshops (Hands-on Practice)

1. **[Basic Setup](workshops/workshop-01-basic-setup.md)**
   - Installing Git on Windows/macOS/Linux
   - Configuring user settings and SSH keys
   - Setting up GitHub account and authentication
   - *Corresponding Tutorials: [Basic Setup](tutorials/05-basic-setup.md) & [Personal Access Tokens](tutorials/07-github-personal-access-tokens.md)*

2. **[First Repository](workshops/workshop-02-first-repo.md)**
   - Creating your first repository
   - Basic Git commands (init, add, commit)
   - Connecting to GitHub remote
   - *Corresponding Tutorial: [Repository Basics](tutorials/02-repository-basics.md)*

3. **[File Operations](workshops/workshop-03-file-operations.md)**
   - Adding, modifying, and deleting files
   - Working with different file types
   - Ignoring files with `.gitignore`
   - *Corresponding Tutorial: [Working with Files](tutorials/03-working-with-files.md) & [Git States](tutorials/04-git-states.md)*

4. **[Branching](workshops/workshop-04-branching.md)**
   - Creating and switching branches
   - Merging branches and resolving conflicts
   - Branch naming conventions and strategies
   - *Corresponding Tutorial: [Branching Fundamentals](tutorials/06-branching-fundamentals.md)*

### 📝 Homework Assignments

1. **[Personal Repository](homeworks/homework-01-personal-repo.md)**
   - Create and configure a personal GitHub repository
   - Set up project structure and documentation
   - Implement basic Git workflow with proper commit practices

2. **[Branching Exercise](homeworks/homework-02-branching-exercise.md)**
   - Practice Git branching and file operations
   - Resolve merge conflicts and manage parallel development
   - Implement feature branch workflow with proper collaboration

3. **[Personal Site with GitHub Pages](homeworks/homework-03-personal-site-github-pages.md)**
   - Create a personal portfolio website using GitHub Pages
   - Deploy static content and configure custom domains
   - Implement CI/CD workflows for automated deployment

### 📋 Assessments

The `assessments/` directory contains:
- Quiz questions covering Git concepts
- Practical assessment rubrics
- Self-assessment checklists

## Resources & References

### 📖 Official Documentation
- [Pro Git Book](https://git-scm.com/book) - Comprehensive Git reference
- [GitHub Docs](https://docs.github.com/en) - Platform documentation
- [Git Documentation](https://git-scm.com/docs) - Official command reference

### 🛠️ Tools & Setup
- [Git Downloads](https://git-scm.com/downloads) - Installers for all platforms
- [GitHub Desktop](https://desktop.github.com/) - GUI client (optional)
- [GitKraken](https://www.gitkraken.com/) - Advanced Git GUI (optional)

### 📚 Additional Learning
- [Learn Git Branching](https://learngitbranching.js.org/) - Interactive Git visualization
- [Git Cheat Sheet](resources/cheatsheet.md) - Quick command reference
- [Useful Links](resources/useful-links.md) - Curated resource collection

## Getting Started

1. **Install Git** following the platform-specific guides in `installation/`
2. **Set up GitHub account** and configure SSH keys
3. **Complete Workshop 1** to verify your installation
4. **Work through tutorials** in numerical order
5. **Practice with workshops** to build hands-on skills
6. **Submit homework assignments** by the due dates

## Key Git Commands Reference

```bash
# Repository setup
git init                    # Initialize new repository
git clone <url>            # Clone existing repository
git remote add origin <url> # Add remote repository

# Basic workflow
git status                 # Check current status
git add <file>             # Stage files for commit
git commit -m "message"    # Commit staged changes
git push origin main       # Push to remote repository
git pull origin main       # Pull latest changes

# Branching
git branch                 # List branches
git branch <name>          # Create new branch
git checkout <name>        # Switch to branch
git merge <branch>         # Merge branch into current

# History & inspection
git log                    # View commit history
git diff                   # Show unstaged changes
git diff --staged          # Show staged changes
```

## Common Git Workflows

### Feature Branch Workflow
1. Create feature branch: `git checkout -b feature/new-feature`
2. Make changes and commit: `git add . && git commit -m "Add new feature"`
3. Push branch: `git push origin feature/new-feature`
4. Create Pull Request on GitHub
5. Merge after review

### Hotfix Workflow
1. Create hotfix branch from main: `git checkout -b hotfix/bug-fix`
2. Fix issue and commit: `git add . && git commit -m "Fix critical bug"`
3. Merge back to main: `git checkout main && git merge hotfix/bug-fix`
4. Delete hotfix branch: `git branch -d hotfix/bug-fix`

## Assessment Criteria

- **Understanding**: Demonstrate comprehension of Git concepts
- **Practical Skills**: Successfully execute Git commands and workflows
- **Problem Solving**: Resolve conflicts and troubleshoot issues
- **Best Practices**: Follow Git conventions and team workflows
- **Documentation**: Clear commit messages and documentation

## Support & Help

- Check the [resources](resources/) directory for additional help
- Review [GitHub Issues](https://github.com/features/issues) for common problems
- Use `git --help <command>` for detailed command information
- Join study groups to discuss challenges and solutions

## Next Steps

After completing this subject, you'll be ready for:
- **Subject 2**: Virtual environments and dependency management
- **Subject 3**: Project management with GitHub Issues and CI/CD
- Advanced Git techniques in future subjects

---

*This subject provides the foundation for all version control activities throughout the course. Mastering these concepts will enable effective collaboration on all subsequent projects.*
