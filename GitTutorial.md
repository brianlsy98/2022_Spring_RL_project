Git tutorial

# initially clone (only 1 time)
git clone https://github.com/brianlsy98/2022_Spring_RL_project.git

# Upload your local codes to remote repository
git add .
git commit –m “commit message”
git push –u origin main		** if error : try pulling first **

# Download remote repository’s codes to local PC
git pull origin main

# make branch
git checkout –b <branchname>
git merge <branchname1> <branchname2>

# undo local changes
git fetch origin
git reset –-hard origin/main
