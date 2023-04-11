#! /bin/zsh

conda env export --name vegi_esc_server --from-history > environment.yml
conda env export --name vegi_esc_server --from-history > predict-environment.yml


conda-lock -f environment.yml -f pot-environment.yml -p linux-64 -k explicit --filename-template "conda-{platform}.lock"
conda-lock -f predict-environment.yml -f pot-environment.yml -p linux-64 -k explicit --filename-template "predict-{platform}.lock"
# conda lock -p osx-64 -p osx-arm64 -p linux-64 -p linux-aarch64 -f environment.yml
# conda lock -p osx-64 -p osx-arm64 -p linux-64 -p linux-aarch64 -f predict-environment.yml --filename-template 'predict-{platform}.lock'