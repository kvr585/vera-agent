# VERA Developer Guide

Welcome to the VERA (Versioned Extensible Robotic Agent) development environment. This guide provides instructions on how to install, build, run, and test the VERA Agent Engine.

---

## 1. Prerequisites

Ensure you have the following installed on your host system:
- **Python 3.12+** (VERA V0.1 utilizes Python 3.14 on this machine)
- **Git**
- **Ollama** (for running local LLM providers)

---

## 2. Environment Setup

VERA utilizes **`uv`** for extremely fast dependency management and virtual environment orchestration.

### Step 2.1: Install `uv`
If `uv` is not globally available on your system, install it using `pip` from your base python installation:
```powershell
python -m pip install uv
```

### Step 2.2: Synchronize Dependencies & Create Virtual Environment
Run the sync command to automatically create the `.venv` directory, install all runtime dependencies, and set up development tool groups:
```powershell
python -m uv sync
```

---

## 3. Running Code Quality Tools

To maintain the high-quality standards defined in the project architecture, verify code changes using `ruff` and `mypy` before committing.

### 3.1: Ruff (Linter and Formatter)
Ruff is used for rapid style checking and import sorting.
```powershell
# Run linting checks
python -m uv run ruff check

# Automatically fix fixable issues
python -m uv run ruff check --fix

# Format code files
python -m uv run ruff format
```

### 3.2: Mypy (Static Type Checking)
VERA mandates strict typing (`disallow_untyped_defs = true`). Run static type analysis using:
```powershell
.venv\Scripts\python.exe -m mypy src
```

> [!WARNING]
> **Windows Application Control Policy Note:**
> In some Windows environments with strict WDAC (Windows Defender Application Control) or AppLocker policies, the precompiled C-extension DLLs shipped inside the default `mypy` wheel may be blocked from loading.
> If you encounter: `ImportError: DLL load failed while importing internal: An Application Control policy has blocked this file`, rely on IDE-level type checking (like Pyright or Pylance in VS Code) and standard `ruff check` during local development.

---

## 4. Running the Test Suite

We use `pytest` for unit and integration tests, with automated code coverage generation.

To execute the test suite, run:
```powershell
python -m uv run pytest
```

This will run all files matching `test_*.py` under the `tests/` directory and print a coverage summary to the console. A detailed HTML coverage report will be generated under `htmlcov/index.html`.

---

## 5. Running the Application CLI

The VERA Engine V0.1 interface is driven by a Typer CLI.
To execute a goal through the engine, boot the application via `uv`:
```powershell
python -m uv run python -m vera_engine.main --goal "Your goal here" --config config/config.yaml
```
