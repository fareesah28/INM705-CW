#!/bin/bash

# Load environment
source /opt/flight/etc/setup.sh
flight env activate gridware
module add compilers/gcc gnu

# Setup pyenv + virtualenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"

if [ ! -d "$PYENV_ROOT" ]; then
    git clone https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
    git clone https://github.com/pyenv/pyenv-virtualenv.git "$PYENV_ROOT/plugins/pyenv-virtualenv"
fi

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Install Python
https_proxy=http://hpc-proxy00.city.ac.uk:3128 \
CPPFLAGS="-I/opt/apps/gnu/include" \
LDFLAGS="-L/opt/apps/gnu/lib -L/opt/apps/gnu/lib64 -ltinfo" \
PYTHON_CONFIGURE_OPTS="--with-openssl=$(brew --prefix openssl)" \
pyenv install -s 3.9.5

pyenv virtualenv -f 3.9.5 inm705_env
pyenv activate inm705_env
echo "inm705_env" > .python-version

# Install packages
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 --upgrade pip
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 -r requirements.txt
pip3 install --proxy http://hpc-proxy00.city.ac.uk:3128 numpy==1.26.4 --force-reinstall

export LD_LIBRARY_PATH=/opt/apps/flight/env/conda+jupyter/lib:$LD_LIBRARY_PATH