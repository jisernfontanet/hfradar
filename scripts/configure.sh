# Parameters

DEFAULT_CONFIG_FILE="config/default.config"     # Default Configuration file
LOCAL_CONFIG_FILE="config/local.config"         # Local configuration file
CODENAME_CODAR="codar_process_radials.py"

# Set the paths

CODE_PATH="$MAIN_PATH/codes"

# Load variables from the configuration file. First search for a local
# file. If it is not found, use the general file.

if [[ -f "$MAIN_PATH/$LOCAL_CONFIG_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$MAIN_PATH/$LOCAL_CONFIG_FILE"
else
    echo "Local configuration file not found: $MAIN_PATH/$LOCAL_CONFIG_FILE"
    if [[ -f "$MAIN_PATH/$DEFAULT_CONFIG_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$MAIN_PATH/$DEFAULT_CONFIG_FILE"
    else
        echo "Default configuration file not found: $MAIN_PATH/$DEFAULT_CONFIG_FILE"
        exit 1
    fi
fi

# There is a problem with paths that has to be solved. Solved locally

if [[ -f "$CONDA_SOURCE_SHELL" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SOURCE_SHELL"
else
  echo "$CONDA_SOURCE_SHELL" not found
fi
