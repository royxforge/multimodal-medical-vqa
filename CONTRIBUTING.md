# Contributing to MedVQA

Thank you for your interest in contributing to MedVQA! This document provides
guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Conventions](#commit-message-conventions)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code. Please report unacceptable behavior to royxforge@gmail.com.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/multimodal-medical-vqa.git
   cd multimodal-medical-vqa
   ```
3. Add the upstream repository:
   ```
   git remote add upstream https://github.com/royxforge/multimodal-medical-vqa.git
   ```

## Development Setup

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for local GPU mode)
- Node.js 20+ (for frontend)
- npm

### Backend Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Download Dataset (optional)

```bash
python scripts/download_vqa_rad.py
```

### Verify Installation

```bash
python -c "from src.inference.pipeline import MedVQAPipeline; print('Backend OK')"
cd frontend && npm run lint
```

## Coding Standards

### Python

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide.
- Use type annotations for all function signatures.
- Use `from __future__ import annotations`.
- Maximum line length: 88 characters (Black-compatible).
- Use descriptive variable names.

### TypeScript/React

- Follow the project's existing ESLint configuration.
- Use TypeScript strict mode.
- Use React functional components with hooks.
- Use Tailwind CSS utility classes.

### Imports

Organize imports in the following order:

1. Standard library imports
2. Third-party imports
3. Local application imports

## Testing

```bash
pytest tests/ -v
pytest tests/test_data.py -v
pytest tests/test_model.py -v
pytest tests/test_inference.py -v
pytest tests/test_training_pipeline.py -v
```

All 49 tests should pass before submitting changes.

## Pull Request Process

1. Create a new branch from `main`:
   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, descriptive commit messages.

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

4. If adding new functionality, add corresponding tests.

5. Push your branch and open a Pull Request on GitHub.

6. In your PR description, include:
   - What the change does
   - Any relevant issue numbers
   - How you tested the change
   - Screenshots if applicable

7. Request review from a maintainer.

## Commit Message Conventions

We follow conventional commit format:

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(fusion): add cross-attention with learned visual type embedding
fix(data): correct VQA-RAD stratified split proportions
docs(readme): update GPT-4o evaluation results
```

## Issue Reporting

### Bug Reports

When filing a bug report, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior and actual behavior
- Environment details (OS, Python version, CUDA version)
- Relevant logs or error messages

### Feature Requests

We welcome feature suggestions! Please include:

- A clear description of the proposed feature
- The clinical or research motivation
- Any relevant references
- Whether you are willing to implement it

Thank you for helping make MedVQA better!
