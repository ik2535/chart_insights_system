# Development Guide

This document outlines the development workflow and practices for the Chart Insights System.

## Development Workflow

We follow a Git Flow-like workflow:

1. **Main Branch**: Contains production-ready code
2. **Develop Branch**: Integration branch for features
3. **Feature Branches**: For developing new features
4. **Release Branches**: For preparing releases
5. **Hotfix Branches**: For emergency fixes

### Branch Naming Convention

- Feature branches: `feature/feature-name`
- Bugfix branches: `bugfix/issue-description`
- Hotfix branches: `hotfix/issue-description`
- Release branches: `release/vX.Y.Z`

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chart_insights_system
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. Set up Neo4j:
   Follow the instructions in the README.md file to set up Neo4j.

5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

1. **Create a new feature branch**:
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/your-feature-name
   ```

2. **Implement your changes**:
   - Write code
   - Write tests
   - Update documentation

3. **Run tests locally**:
   ```bash
   # Run all tests
   pytest
   
   # Run unit tests only
   pytest tests/unit
   
   # Run with coverage
   pytest --cov=src tests/
   ```

4. **Format your code**:
   ```bash
   black .
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```
   
   We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test changes
   - `refactor:` for refactoring
   - `chore:` for maintenance tasks

6. **Push your branch**:
   ```bash
   git push -u origin feature/your-feature-name
   ```

7. **Create a Pull Request**:
   - Go to the repository on GitHub
   - Create a new Pull Request from your branch to `develop`
   - Fill in the PR template
   - Request a review

8. **Address review comments**:
   - Make necessary changes
   - Push additional commits
   - Update the PR

9. **Merge to develop**:
   - The PR will be merged to `develop` after approval
   - CI/CD will run tests automatically

## Testing Guidelines

1. **Write tests first**: Follow Test-Driven Development (TDD) when possible.
2. **Test coverage**: Aim for at least 80% test coverage.
3. **Test types**:
   - **Unit tests**: Test individual components in isolation
   - **Integration tests**: Test interactions between components
   - **End-to-end tests**: Test the complete flow
4. **Test organization**:
   - Unit tests go in `tests/unit/`
   - Integration tests go in `tests/integration/`
   - Fixtures and conftest go in `tests/conftest.py`
5. **Mocking**:
   - Use mocks for external services (Neo4j, LLMs, etc.)
   - Use the fixtures defined in `conftest.py` wherever possible

## Code Style

We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide with some modifications:

1. **Line length**: 100 characters
2. **Formatting**: We use [Black](https://black.readthedocs.io/) for automatic formatting
3. **Documentation**: Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
4. **Type hints**: Use type hints wherever possible

## Documentation

1. **Code documentation**:
   - Every module, class, and function should have docstrings
   - Use Google-style docstrings

2. **Project documentation**:
   - User guides go in `docs/user/`
   - Developer guides go in `docs/dev/`
   - API documentation is generated from docstrings

3. **Updating documentation**:
   - Update documentation when making code changes
   - Build documentation locally to check for errors:
     ```bash
     mkdocs build
     ```
   - Preview documentation locally:
     ```bash
     mkdocs serve
     ```

## Versioning

We follow [Semantic Versioning](https://semver.org/):

1. **Major version** (X.0.0): Breaking changes
2. **Minor version** (0.X.0): New features without breaking changes
3. **Patch version** (0.0.X): Bug fixes and minor changes

## Release Process

1. **Create a release branch**:
   ```bash
   git checkout develop
   git pull
   git checkout -b release/vX.Y.Z
   ```

2. **Prepare for release**:
   - Update version numbers
   - Update CHANGELOG.md
   - Final testing and fixes

3. **Create a Pull Request**:
   - Create a PR from `release/vX.Y.Z` to `main`
   - Get approval from team members

4. **Merge to main**:
   - Merge the PR to `main`
   - Tag the release:
     ```bash
     git checkout main
     git pull
     git tag -a vX.Y.Z -m "Release vX.Y.Z"
     git push origin vX.Y.Z
     ```

5. **Merge back to develop**:
   - Create a PR from `main` to `develop`
   - Resolve any conflicts
   - Merge the PR

## Continuous Integration/Continuous Deployment (CI/CD)

We use GitHub Actions for CI/CD:

1. **Continuous Integration**:
   - Runs on every push to `main` and `develop`, and on all PRs
   - Runs tests, linting, and coverage
   - Ensures code quality

2. **Continuous Deployment**:
   - Builds and deploys documentation on merges to `main`
   - Builds and publishes packages on new tags

## Issue Tracking

We use GitHub Issues for tracking work:

1. **Issue types**:
   - **Bug**: Something isn't working
   - **Feature**: New feature request
   - **Enhancement**: Improvement to existing features
   - **Documentation**: Documentation-related tasks
   - **Technical Debt**: Code improvements, refactoring

2. **Issue workflow**:
   - **New**: Issue has been created
   - **Triage**: Issue needs assessment
   - **Backlog**: Issue is prioritized but not being worked on
   - **In Progress**: Issue is being worked on
   - **Review**: Solution needs review
   - **Done**: Issue is completed

## Development Environment

### Recommended Tools

1. **Code Editor**: VS Code with Python extension
2. **Database GUI**: Neo4j Desktop
3. **API Testing**: Postman or Insomnia
4. **Virtual Environment**: venv or conda

### VS Code Extensions

- Python
- Pylance
- Python Docstring Generator
- GitLens
- Git Graph
- YAML
- Markdown All in One

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.
