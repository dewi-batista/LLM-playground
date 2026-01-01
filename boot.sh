####==== to make new virtual environment
# python3.12 -m venv .venv
# pip install pipreqs
# pip freeze > requirements.txt
# pipreqs . --force --ignore .venv
####====

# load modules + virtual environment
module load Python/3.12.3
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate

# make pip, requirements.txt and repo are up-to-date
pip install --upgrade pip
git yem

# tmux new -s train
# watch -n 2.5 'df -h .; echo; df -ih .'
# watch -n 2.5 nvidia-smi