#!/bin/bash

# Load environment
source /opt/flight/etc/setup.sh
flight env activate gridware
module add compilers/gcc gnu

# Setup pyenv + virtualenv paths
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

# Clone pyenv and pyenv-virtualenv if not present
if [ ! -d "$PYENV_ROOT" ]; then
    git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
    git clone https://github.com/pyenv/pyenv-virtualenv.git "$PYENV_ROOT/plugins/pyenv-virtualenv"
fi

# Init pyenv
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python 3.9.5 with SSL + libffi fixes
https_proxy=http://hpc-proxy00.city.ac.uk:3128 \
CPPFLAGS="-I/opt/apps/gnu/include" \
LDFLAGS="-L/opt/apps/gnu/lib -L/opt/apps/gnu/lib64 -ltinfo" \
PYTHON_CONFIGURE_OPTS="--with-openssl=$(brew --prefix openssl)" \
pyenv install -s 3.9.5

# Create and activate virtualenv
pyenv virtualenv -f 3.9.5 inm705_env
pyenv activate inm705_env
echo "inm705_env" > .python-version

# Upgrade pip and install build tools
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 --upgrade pip
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 scikit-build cmake

# Install required packages
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 "transformers>=4.31.0" huggingface-hub torch tensorflow==2.12.0
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 -r requirements.txt
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 numpy==1.26.4 --force-reinstall

# Fix for SSL and libffi shared library issues at runtime
export LD_LIBRARY_PATH=/opt/apps/flight/env/conda+jupyter/lib:$LD_LIBRARY_PATH