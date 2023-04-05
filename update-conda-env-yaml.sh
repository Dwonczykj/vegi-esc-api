#! /bin/zsh

conda env export --name vegi-esc-server --from-history > environment.yml
conda env export --name vegi-esc-server --from-history > predict-environment.yml

conda lock -p osx-arm64 -p linux-64 -f environment.yml #-k explicit --filename-template '{platform}.lock'
conda lock -p osx-arm64 -p linux-64 -f predict-environment.yml -k explicit --filename-template 'predict-{platform}.lock'
# conda lock -p osx-64 -p osx-arm64 -p linux-64 -p linux-aarch64 -f environment.yml
# conda lock -p osx-64 -p osx-arm64 -p linux-64 -p linux-aarch64 -f predict-environment.yml --filename-template 'predict-{platform}.lock'