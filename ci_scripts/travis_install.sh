#!/bin/bash

set -e
echo 'Start time'
date '+%Y-%m-%Y %H:%M:%S'
echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip
echo 'ccache:'
ls $HOME/.ccache

if [[ "$DISTRIB" == "conda" ]]; then
    echo $DISTRIB
    echo 'Setting up a conda-based virtual environment'
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    # Use the miniconda installer for faster download / install of conda
    # itself
    DOWNLOAD_DIR=${DOWNLOAD_DIR:-$HOME/.tmp/miniconda}
    mkdir -p $DOWNLOAD_DIR
    wget http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh \
        -O $DOWNLOAD_DIR/miniconda.sh
    chmod +x $DOWNLOAD_DIR/miniconda.sh && \
        bash $DOWNLOAD_DIR/miniconda.sh -b -p $HOME/miniconda && \
        rm -r -d -f $DOWNLOAD_DIR
    export PATH=$HOME/miniconda/bin:$PATH
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    conda create -n testenv --yes python=$PYTHON_VERSION pip
    source activate testenv
fi


# Install all required packages ('hottbox' and testing dependencies)
REQUIREMENTS_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
pip install -r $REQUIREMENTS_HOME/travis_requirements.txt

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi