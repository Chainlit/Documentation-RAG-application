#!/bin/bash

SCRIPT=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

rm -rf ./documentation
mkdir -p ./documentation
git clone git@github.com:Chainlit/literal-docs.git

FILE_PATHS=$(find ./literal-docs -name "*.mdx")

while IFS= read -r file; do
    cp "$file" ./documentation/
done <<< "$FILE_PATHS"

rm -rf ./literal-docs

python3 "$SCRIPT_DIR/main.py"
