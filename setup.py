#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read version from __init__.py
version = {}
with open("src/repo_intel/__init__.py") as f:
    exec(f.read(), version)

setup(
    name="repo-intel",
    version=version["__version__"],
    description="Repository Intelligence Tools - Analyze codebases with LLM assistance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Greg Doermann",
    author_email="gdoermann@gmail.com",
    url="https://github.com/gdoermann/repo-intel",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "attrdict3>=2.0.0",
        "boto3>=1.26.0",
        "requests>=2.28.0",
        "environs>=9.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "repo-intel=repo_intel.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Version Control :: Git",
        "Topic :: Text Processing :: Markup :: Markdown",
    ],
    keywords="git analysis llm code-review repository intelligence aws-glue markdown",
    project_urls={
        "Bug Reports": "https://github.com/gdoermann/repo-intel/issues",
        "Source": "https://github.com/gdoermann/repo-intel",
        "Documentation": "https://github.com/gdoermann/repo-intel#readme",
    },
)
