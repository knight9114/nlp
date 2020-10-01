# -------------------------------------------------------------------------
#   Install Script
# -------------------------------------------------------------------------
# Imports
from setuptools import setup, find_packages

# Load Read-Me
with open("README.md", "r") as fp:
    long_description = fp.read()

# Create Setup
setuptools.setup(
    name="nlp",
    version="0.0.1",
    author="Knight9114",
    description="NLP implementations in PyTorch, Tensorflow, and MXNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/knight9114/nlp",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
