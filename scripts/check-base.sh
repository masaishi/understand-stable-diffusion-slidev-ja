#!/bin/bash

# Define the file and pattern
FILE="./vite.config.ts"
PATTERN="//base: '/understand-stable-diffusion-slidev-ja/',"
CORRECT_PATTERN="base: '/understand-stable-diffusion-slidev-ja/',"

# Check if the file contains the specific commented line
if grep -q "$PATTERN" "$FILE"; then
    echo "Warning: Base URL is commented."
    # Uncomment the line using sed with compatibility for macOS and Linux
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS requires an empty string argument with -i for in-place editing
        sed -i '' "s|$PATTERN|$CORRECT_PATTERN|g" "$FILE"
    else
        # Linux does not require an empty string argument
        sed -i "s|$PATTERN|$CORRECT_PATTERN|g" "$FILE"
    fi
    echo "The line has been uncommented."
else
    echo "No changes made. The line does not exist or is already uncommented."
fi
