#!/bin/bash

# this script is run by jina-dev-bot

# update autocomplete info && black it
python generate-docstring.py && black -S ../discoart/create.py