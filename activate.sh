#!/bin/bash

# Set the name of the virtual environment directory
VENV_DIR="venv"

# Check if the virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' does not exist. Creating it..."
    python -m venv "$VENV_DIR"
    echo "Virtual environment '$VENV_DIR' created."
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# Activate the virtual environment

# For Linux/Mac
source "$VENV_DIR/bin/activate"


# Confirm the virtual environment is activated
echo "Virtual environment '$VENV_DIR' is now activated."