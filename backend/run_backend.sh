#!/bin/bash

# Ensure script fails on error
set -e

# Confirm that uvicorn is installed
if ! command -v uvicorn &> /dev/null
then
    echo "Error: Uvicorn is not installed."
    exit 1
fi

# Start Backend (FastAPI)
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
