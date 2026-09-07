# Contributing to Crypto Portfolio System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the problem
- Expected behavior vs actual behavior
- Your environment (OS, Python version, package version)
- Any relevant logs or error messages

### Suggesting Features

Feature suggestions are welcome. Please provide:

- A clear description of the proposed feature
- The motivation or use case
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Add or update tests as needed
5. Ensure all tests pass
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Setup

```bash
git clone https://github.com/sachncs/optimising-cryptocurrency-portfolios.git
cd optimising-cryptocurrency-portfolios
pip install -e ".[dev]"
```

### Running Tests

```bash
PYTHONPATH=src pytest -q
```

### Running with Coverage

```bash
PYTHONPATH=src python -m coverage run -m pytest -q
python -m coverage report -m
```

## Branch Naming

Use descriptive branch names:

- `feat/short-description` for new features
- `fix/short-description` for bug fixes
- `docs/short-description` for documentation changes
- `refactor/short-description` for code refactoring

## Commit Conventions

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new forecasting method
fix: resolve edge case in portfolio optimization
docs: update API reference
refactor: simplify correlation matrix computation
test: add tests for governance module
chore: update dependencies
```

## Code Standards

- Follow PEP 8 for Python code
- Use type hints where possible
- Write docstrings for public functions and classes
- Keep functions focused and reasonably sized
- Add tests for new functionality

## Pull Request Process

1. Update documentation if your change affects the API or user-facing behavior
2. Add entries to CHANGELOG.md under `[Unreleased]`
3. Ensure CI passes
4. Request review from maintainers

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold its standards.

## Security

If you discover a security vulnerability, please follow the process outlined in [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
