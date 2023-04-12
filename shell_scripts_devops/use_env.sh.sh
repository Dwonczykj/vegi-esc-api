#! /bin/zsh

my_new_env=word2vec_pip_env

# echo "Comment the line below to skip creating the virtual pip env"
# python3 -m venv $my_new_env

source $my_new_env/bin/activate
which python # /Users/joey/Github_Keep/app/word2vec_pip_env/bin/python
# python3 -m pip install -r requirements.txt

# echo "Uncomment the line below to deactivate the virtual pip env"
# deactivate