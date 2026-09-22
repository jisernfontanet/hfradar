#!/usr/bin/env bash

# Get the location of the script's main path

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PATH="$(dirname "$SCRIPT_PATH")"

# Configure paths and libraries

source "$MAIN_PATH/scripts/configure.sh"    # Configure paths
conda activate "$CONDA_ENV"                 # Activate the CONDA environment

# Run the code

echo "$CODENAME_CODAR"
echo "$CONDA_SOURCE_SHELL"
echo "$CONDA_ENV"
#python "$CODE_PATH/$CODENAME_CODAR"


