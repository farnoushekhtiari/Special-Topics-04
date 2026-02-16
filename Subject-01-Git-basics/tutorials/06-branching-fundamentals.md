# Tutorial 6: Git Branching Fundamentals

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand the concept and purpose of Git branches
- Explain how branches enable parallel development
- Create, switch between, and manage branches
- Visualize branch relationships and history
- Apply branching strategies for different development scenarios
- Understand branch naming conventions and best practices

## Prerequisites
- Git installed and configured (see [Tutorial 5](05-basic-setup.md))
- Basic repository operations (see [Tutorial 2](02-repository-basics.md))
- Understanding of commits and file states (see [Tutorial 3](03-working-with-files.md))

## What is a Branch?

A **Git branch** is a lightweight, movable pointer to a commit. Branches allow you to:
- **Isolate development**: Work on features without affecting main code
- **Parallel workflows**: Multiple team members work simultaneously
- **Safe experimentation**: Try ideas without risking stable code
- **Organized releases**: Manage different versions and releases

### Branch vs. Copy
Unlike copying files, branches in Git are:
- **Efficient**: Share storage, only track differences
- **Fast**: Creating/switching branches is instantaneous
- **Safe**: Changes on one branch don't affect others automatically
- **Mergeable**: Can combine branches when ready

## Part 1: Understanding Branch Structure

### The Commit Graph

Git's history forms a **directed acyclic graph (DAG)**:
```
A ── B ── C ── D  (main branch)
            │
            └─ E ── F  (feature branch)
```

- **Commits**: Snapshots of your project at points in time
- **Branches**: Named pointers to specific commits
- **HEAD**: Special pointer showing your current location

### Branch Pointers

```
main ────► C (commit)
           │
feature ──► F (commit)
           │
HEAD ─────► feature (currently on feature branch)
```

## Part 2: Basic Branch Operations

### Viewing Branches

```bash
# List all local branches
git branch

# List all branches (local and remote)
git branch -a

# Show current branch with additional info
git branch -v

# Show branch hierarchy
git log --oneline --graph --all --decorate
```

### Creating Branches

```bash
# Create and stay on current branch
git branch feature-login

# Create and switch to new branch
git switch -c feature-dashboard

# Alternative syntax (older)
git checkout -b bug-fix
```

### Switching Between Branches

```bash
# Switch to existing branch
git switch main
git switch feature-login

# Alternative syntax (older)
git checkout main
git checkout feature-login
```

### Branch Creation Patterns

**From Current HEAD:**
```bash
git switch -c new-feature  # Creates from current commit
```

**From Specific Commit:**
```bash
git switch -c hotfix abc123  # Creates from specific commit hash
```

**From Remote Branch:**
```bash
git switch -c feature-remote origin/feature-remote
```

## Part 3: Working with Branches

### Branch Isolation

Each branch maintains its own:
- **Commit history**: Independent sequence of changes
- **File states**: Different versions of files
- **Development pace**: Features can progress independently

**Example Workflow:**
```bash
# Start on main
git switch main
git status  # Clean working directory

# Create feature branch
git switch -c feature/user-profile

# Make changes (only on this branch)
echo "User profile functionality" > user_profile.py
git add user_profile.py
git commit -m "Add user profile module"

# Switch back to main
git switch main
ls  # user_profile.py is gone!
```

### Branch Status

```bash
# Show current branch and status
git status

# Show branch information
git branch -v

# Show relationship to remote branches
git status -b
```

## Part 4: Branch History and Visualization

### Viewing Branch Relationships

```bash
# Simple branch graph
git log --oneline --graph --all

# Detailed branch view
git log --oneline --graph --all --decorate

# Branch-specific history
git log --oneline feature-branch

# Compare branches
git log --oneline main..feature-branch  # What’s in feature not in main
git log --oneline feature-branch..main  # What’s in main not in feature
```

### Understanding Branch Divergence

```bash
# Show commits unique to each branch
git log --oneline --graph main...feature-branch

# Show merge base (common ancestor)
git merge-base main feature-branch
```

## Part 5: Branch Management

### Renaming Branches

```bash
# Rename current branch
git branch -m new-feature-name

# Rename any branch (not current)
git branch -m old-name new-name
```

### Deleting Branches

```bash
# Safe delete (only if merged)
git branch -d feature-completed

# Force delete (even if not merged)
git branch -D feature-abandoned

# Delete remote branch
git push origin --delete feature-old
```

### Tracking Remote Branches

```bash
# Set upstream for new branch
git push -u origin feature-new

# Track existing remote branch
git branch --set-upstream-to=origin/main

# View tracking relationships
git branch -vv
```

## Part 6: Branching Strategies and Best Practices

### Common Branching Patterns

#### 1. Feature Branches
```bash
# For new features
git switch -c feature/user-authentication
# Work on feature
git switch main
git merge feature/user-authentication
git branch -d feature/user-authentication
```

#### 2. Bug Fix Branches
```bash
# For bug fixes
git switch -c fix/login-validation
# Fix the bug
git switch main
git merge fix/login-validation
git branch -d fix/login-validation
```

#### 3. Release Branches
```bash
# Prepare for release
git switch -c release/v1.0
# Final testing and fixes
git switch main
git merge release/v1.0
git tag v1.0
```

#### 4. Hotfix Branches
```bash
# Emergency fixes on production
git switch -c hotfix/critical-bug main  # Branch from main
# Fix the bug
git switch main
git merge hotfix/critical-bug
git switch production  # Assuming production branch exists
git merge hotfix/critical-bug
```

### Branch Naming Conventions

**Feature Branches:**
```
feature/user-login
feature/payment-integration
feature/dark-mode-ui
```

**Bug Fix Branches:**
```
fix/login-validation
fix/memory-leak
fix/ui-responsive
```

**Release Branches:**
```
release/v2.1.0
release/2024-q1
```

**Hotfix Branches:**
```
hotfix/security-patch
hotfix/critical-bug
```

### Branch Protection Rules

**Main/Master Branch:**
- Require pull requests for merges
- Require status checks (CI/CD)
- Require code reviews
- Prevent force pushes

**Development/Release Branches:**
- Require pull requests
- Allow some direct pushes from maintainers
- Require tests to pass

## Part 7: Advanced Branch Concepts

### Detached HEAD State

When HEAD points directly to a commit instead of a branch:

```bash
# Enter detached HEAD
git switch abc123  # Switch to commit hash

# You're now in detached HEAD state
git branch  # No current branch shown

# Create branch to exit detached HEAD
git switch -c temp-branch
```

**When this happens:**
- You're viewing historical state
- Changes can be lost if not committed to a branch
- Useful for inspecting old code

### Branch Rebasing

Advanced topic - covered in future tutorials:
```bash
# Rebase feature onto main (linear history)
git switch feature-branch
git rebase main
```

### Remote Branch Management

```bash
# Fetch all remote branches
git fetch --all

# Sync local branches with remote
git pull --all

# Clean up merged remote branches
git remote prune origin
```

## Part 8: Troubleshooting Branch Issues

### "Branch already exists"
```bash
# Check existing branches
git branch

# Delete if needed
git branch -D branch-name

# Or use different name
git switch -c branch-name-v2
```

### "Cannot delete branch"
```bash
# Branch not fully merged
git branch -D branch-name  # Force delete

# Or merge first
git merge branch-name
git branch -d branch-name
```

### Lost Commits After Branch Delete
```bash
# Find lost commits
git reflog

# Recreate branch from commit
git branch recovered-branch abc123
```

### Branch Divergence Issues
```bash
# See what differs
git log --oneline main..feature
git log --oneline feature..main

# Merge or rebase to resolve
git merge feature  # or git rebase main
```

## Part 9: Branch Workflows in Teams

### Centralized Workflow
- Single main branch
- All developers work directly on main
- Simple but risky for larger teams

### Feature Branch Workflow
- Developers create feature branches
- Pull requests for code review
- Merge to main after approval

### Git Flow
- main (production releases)
- develop (integration branch)
- feature/ (feature development)
- release/ (release preparation)
- hotfix/ (emergency fixes)

### GitHub Flow
- main (deployable state)
- feature branches (everything else)
- Pull requests for all changes
- Continuous deployment from main

## Summary

✅ **Branch Creation:** `git switch -c <name>` or `git branch <name>`
✅ **Branch Switching:** `git switch <name>`
✅ **Branch Visualization:** `git log --oneline --graph --all`
✅ **Branch Management:** Create, rename, delete branches safely
✅ **Branch Strategies:** Feature branches, bug fixes, releases
✅ **Best Practices:** Naming conventions, protection rules
✅ **Troubleshooting:** Handle common branch issues

## Key Branching Concepts

1. **Isolation:** Branches keep work separate and safe
2. **Parallel Development:** Multiple features can progress simultaneously
3. **Merge Integration:** Combine completed work back to main branch
4. **History Tracking:** Clear lineage of changes and features
5. **Collaboration:** Enable team workflows and code review

## Common Branch Commands

```bash
# Basic operations
git branch                    # List branches
git branch -v                 # Verbose branch info
git switch <branch>           # Switch branch
git switch -c <name>          # Create and switch

# Advanced viewing
git log --oneline --graph --all    # Branch relationships
git branch -vv                 # Tracking info

# Management
git branch -m <new-name>       # Rename branch
git branch -d <branch>         # Safe delete
git branch -D <branch>         # Force delete
```

## Next Steps

Now that you understand branching:
- [Practice with Workshop 4](../workshops/workshop-04-branching.md)
- [Learn about merging](../tutorials/07-merging-strategies.md)
- [Explore team workflows](../tutorials/08-collaboration-workflows.md)
- [Set up protected branches](../tutorials/09-branch-protection.md)

## Additional Resources

- [Git Branching Documentation](https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell)
- [Branching Strategies](https://www.atlassian.com/git/tutorials/comparing-workflows)
- [GitHub Flow Guide](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Branch Naming Conventions](https://stackoverflow.com/questions/273695/git-branch-naming-best-practices)

---

*Mastering Git branches unlocks powerful development workflows. Start with feature branches for your next project!*