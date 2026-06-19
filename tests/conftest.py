# tests/conftest.py
# Ensure Kaggle credentials are present during pytest collection so importing
# the kaggle package doesn't call exit() in environments without real creds.
# These are safe defaults for CI and local test runs; real credentials should
# be provided via GitHub Secrets if you need real Kaggle access.
import os

os.environ.setdefault("KAGGLE_USERNAME", "test")
os.environ.setdefault("KAGGLE_KEY", "test")
