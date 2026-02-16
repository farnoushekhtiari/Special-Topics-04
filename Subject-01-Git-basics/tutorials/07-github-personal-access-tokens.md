# Tutorial 7: GitHub Personal Access Tokens

## Learning Objectives
By the end of this tutorial, you will be able to:
- Understand what GitHub Personal Access Tokens are and why they're needed
- Generate a Personal Access Token with appropriate permissions
- Use Personal Access Tokens for Git operations
- Troubleshoot authentication issues with GitHub
- Manage and revoke access tokens securely

## Prerequisites
- GitHub account (see [Tutorial 5](05-basic-setup.md))
- Basic understanding of Git and version control
- Web browser for GitHub access

## The Problem: Password Authentication Removed

In 2021, GitHub removed support for password-based authentication when using Git operations over HTTPS. This security measure was implemented to protect user accounts and encourage more secure authentication methods.

**Error you'll encounter without PAT:**
```
Support for password authentication was removed. Please use a personal access token instead.
```

## What is a GitHub Personal Access Token?

A **Personal Access Token (PAT)** is a long string of characters that serves as an alternative to password authentication for GitHub. It's essentially an "app password" that you can generate and use specifically for Git operations.

### Key Features of Personal Access Tokens

- **Unique**: Each token is unique to your account
- **Secure**: More secure than using your actual password
- **Controlled**: You can limit what the token can access
- **Revocable**: You can delete tokens at any time
- **Expirable**: You can set expiration dates for security
- **Scoped**: You control which permissions the token has

## Why Use Personal Access Tokens?

1. **Security**: Don't expose your actual GitHub password
2. **Granular Control**: Limit what each token can do
3. **Revocation**: Easy to disable compromised tokens
4. **Auditing**: Track which applications are using your account
5. **Automation**: Use different tokens for different purposes

## How to Create a GitHub Personal Access Token

### Step 1: Access Your GitHub Profile
Click on your profile picture in the top-right corner of any GitHub page.

### Step 2: Navigate to Settings
From the dropdown menu, select **"Settings"**.

### Step 3: Access Developer Settings
Scroll down to the bottom of the left sidebar and click **"Developer settings"**.

### Step 4: Select Personal Access Tokens
In the left sidebar, click **"Personal access tokens"**.

### Step 5: Choose Token Type
Click **"Generate new token"** (choose "Generate new token (classic)" for broader compatibility).

### Step 6: Authenticate
Enter your GitHub password when prompted to confirm access.

### Step 7: Add a Descriptive Note
Give your token a clear, descriptive name (e.g., "Git Operations", "Course Project", "CI/CD Pipeline").

### Step 8: Set Expiration Date
Choose when you want the token to expire:
- **Recommended**: Set a reasonable expiration date (30, 60, or 90 days)
- **Not recommended**: "No expiration" (for security reasons)

### Step 9: Select Appropriate Scopes
Choose the permissions your token needs. For basic Git operations:

**Required scopes for Git operations:**
- ✅ `repo` - Full control of private repositories
- ✅ `public_repo` - Access public repositories (if you work with public repos)

**Additional scopes you might need:**
- ✅ `workflow` - Update GitHub Action workflows
- ✅ `read:org` - Read org membership (if working with organizations)

**Avoid selecting unnecessary permissions for security!**

### Step 10: Generate the Token
Click the **"Generate token"** button at the bottom.

### Step 11: Copy and Store the Token
**Important**: Copy the token immediately! GitHub will only show it once.
- Store it securely (password manager, secure note)
- Never commit tokens to version control
- Never share tokens publicly

### Step 12: Using Your Personal Access Token

Now that you have your PAT, here's how to use it for Git operations:

#### For Cloning Repositories
When cloning a repository that requires authentication:

```bash
# Instead of: git clone https://github.com/username/repo.git
git clone https://YOUR_USERNAME:YOUR_PAT@github.com/YOUR_USERNAME/repo.git

# Example:
git clone https://johnDoe:ghp_1234567890abcdef@github.com/johnDoe/my-project.git
```

#### For Adding Remote Origins
When adding a remote to an existing repository:

```bash
# Instead of: git remote add origin https://github.com/username/repo.git
git remote add origin https://YOUR_USERNAME:YOUR_PAT@github.com/YOUR_USERNAME/repo.git
```

#### For Existing Repositories
If you have an existing repository with authentication issues:

```bash
# Update the remote URL
git remote set-url origin https://YOUR_USERNAME:YOUR_PAT@github.com/YOUR_USERNAME/repo.git
```

**Important Notes:**
- Replace `YOUR_USERNAME` with your actual GitHub username
- Replace `YOUR_PAT` with your personal access token
- The format is: `https://username:token@github.com/username/repo.git`
- This authentication method works for both push and pull operations

## Managing Your Personal Access Tokens

### Viewing Active Tokens
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. See all your active tokens and their expiration dates

### Regenerating Tokens
If you suspect a token is compromised:
1. Delete the old token
2. Generate a new one with the same scopes
3. Update any applications using the old token

### Deleting Tokens
1. Find the token in your list
2. Click the "Delete" button
3. Confirm deletion

## Best Practices for Personal Access Tokens

### Security Best Practices
- **Use descriptive names**: Know what each token is for
- **Set expiration dates**: Don't create tokens that never expire
- **Use minimal scopes**: Only grant necessary permissions
- **Regular rotation**: Regenerate tokens periodically
- **Monitor usage**: Check GitHub's security log for token usage

### Organization Best Practices
- **Separate tokens**: Use different tokens for different projects
- **Team coordination**: Coordinate with team members on token usage
- **Documentation**: Keep track of which tokens are used where

### Storage Best Practices
- **Secure storage**: Use password managers or secure credential storage
- **Environment variables**: Store tokens as environment variables in scripts
- **Never in code**: Don't hardcode tokens in your source code
- **Separate configs**: Use `.git-credentials` or credential helpers securely

## Troubleshooting Common Issues

### "Authentication failed" Error
- Verify your username is correct
- Check that your PAT hasn't expired
- Ensure the token has the correct scopes
- Try regenerating the token

### "Repository not found" Error
- Check repository permissions
- Verify the repository URL is correct
- Ensure your PAT has access to the repository

### Token Suddenly Stops Working
- Check if the token has expired
- Verify it hasn't been revoked
- Confirm GitHub hasn't disabled the token for security reasons

### SSH vs HTTPS Confusion
- PATs only work with HTTPS URLs
- For SSH, use SSH keys instead (see Tutorial 5)

## Advanced Token Management

### Using Git Credential Helper
Store your PAT securely using Git's credential helper:

```bash
# Store credentials (you'll be prompted for username and PAT)
git config --global credential.helper store

# Or use a more secure credential manager
git config --global credential.helper manager  # Windows
git config --global credential.helper osxkeychain  # macOS
```

### Environment Variables in Scripts
For automation scripts, use environment variables:

```bash
# Set environment variable
export GITHUB_TOKEN=your_personal_access_token

# Use in scripts
git clone https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/repo.git
```

### GitHub CLI Integration
If you use GitHub CLI:

```bash
# Authenticate with GitHub CLI
gh auth login

# GitHub CLI will handle authentication automatically
git clone https://github.com/username/repo.git
```

## Summary

✅ **Generated PAT**: Created a personal access token with appropriate scopes
✅ **Configured Authentication**: Set up Git to use PAT for HTTPS operations
✅ **Security Practices**: Implemented token management best practices
✅ **Troubleshooting**: Can resolve common authentication issues
✅ **Advanced Usage**: Understand credential helpers and automation

## Key Takeaways

1. **PATs replace passwords** for GitHub authentication
2. **Minimal scopes** provide better security
3. **Regular rotation** keeps your account secure
4. **Proper storage** prevents token exposure
5. **Correct syntax**: `https://username:token@github.com/username/repo.git`

## Next Steps

Now that you can authenticate with GitHub:
- [Clone repositories securely](../workshops/workshop-02-first-repo.md)
- [Set up your development workflow](../workshops/workshop-01-basic-setup.md)
- [Learn about collaborative workflows](../tutorials/08-collaboration-workflows.md)

## Additional Resources

- [GitHub PAT Documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)
- [Token Scopes Reference](https://docs.github.com/en/developers/apps/building-oauth-apps/scopes-for-oauth-apps)
- [Git Credential Storage](https://git-scm.com/book/en/v2/Git-Tools-Credential-Storage)
- [GitHub CLI](https://cli.github.com/)

---

*Personal Access Tokens are essential for modern GitHub authentication. Use them wisely and keep them secure!*