"""Setup script for the MedVQA package.

Usage:
    pip install -e .   # Editable install for development
"""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="medvqa",
    version="0.1.0",
    description="Multimodal Medical Visual Question Answering System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MedVQA Research",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
