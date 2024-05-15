#!/bin/bash

SCRIPT=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")

rm -rf ./cookbooks ./documentation
mkdir -p ./documentation ./documentation

git clone git@github.com:Chainlit/literal-docs.git
git clone git@github.com:Chainlit/literal-cookbook.git

# Use *.mdx from documentation and README.md, *.py from cookbook
find ./literal-docs -name "*.mdx" -exec bash -c 'newname="./documentation/$(echo {} | sed "s|/|_|g")"; cp "{}" "$newname"' \;
find ./literal-cookbook \( -name "README.md" -o -name "*.py" -o -name "*.ts" \) -exec bash -c 'newname="./cookbooks/$(echo {} | sed "s|/|_|g")"; cp "{}" "$newname"' \;

rm -rf ./literal-docs
rm -rf ./literal-cookbook

python3 "$SCRIPT_DIR/main.py"
