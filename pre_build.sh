#!/bin/bash
# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
