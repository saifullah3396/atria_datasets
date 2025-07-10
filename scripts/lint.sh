#!/usr/bin/env bash

set -e
set -x

# Use provided directory or default to 'src'
DIR=${1:-src}

mypy "$DIR"           # type check
ruff check "$DIR"     # linter
ruff format "$DIR" --check # formatter
