# SST — Sea Surface Temperature & ENSO Prediction (Workshop Version)

This repository is a **minimal, workshop-focused example** that demonstrates how to use **GitHub Actions for CI/CD** in a scientific Python project.

The scientific task—predicting ENSO from Sea Surface Temperature (SST) data using a simple machine learning model—is intentionally lightweight. The goal is **not** to teach climate science or machine learning in depth, but to provide a concrete, realistic workflow that makes CI/CD concepts easy to see and reason about.

Workshop materials: [https://chicago-aiscience.github.io/workshop-github-actions-l1/](https://chicago-aiscience.github.io/workshop-github-actions-l1/)

---

## How This Repo Fits Into the Workshop

This repository is a **simplified derivative** of the full example project:
[https://github.com/chicago-aiscience/workshop-sst](https://github.com/chicago-aiscience/workshop-sst)

The full repository includes more automation, configuration, and production-style structure. For a first CI/CD workshop, that complexity can get in the way. This version exists to:

* Reduce cognitive load for first-time GitHub Actions users
* Highlight *core* CI/CD ideas (triggers, jobs, steps, artifacts)
* Provide fast feedback during hands-on exercises
* Keep workflows short, readable, and easy to modify

Everything you learn here maps directly to the patterns used in the full project.

---

## What the Code Does (At a Glance)

The project:

* Loads sample SST and ENSO (Niño 3.4) CSV data from `data/`
* Applies simple transformations (joins, rolling means, lag features)
* Trains a small Random Forest model
* Writes outputs (predictions, plots, feature importance) to disk

All of this runs in a few seconds on a laptop and inside CI.

---

## Quick Start (Local)

You do **not** need to understand every command below for the workshop—they are provided so you can run the same steps locally that CI runs automatically.

### Install dependencies (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev
```

### Run the workflow end to end

```bash
uv run sst
```

This generates output files in an `artifacts/` directory.

---

## Tests

Run the test suite:

```bash
pytest -q
```

---

## CI/CD (What Matters for the Workshop)

This repository uses **GitHub Actions** to automatically run checks when you push code or open a pull request.

The workflow lives in:

```
.github/workflows/deploy.yml
```

### What the CI Pipeline Does

At a high level, the workflow:

1. Sets up Python
2. Installs dependencies
3. Runs linting and formatting checks
4. Runs tests
5. Produces artifacts

That’s it. Each step is deliberately simple so you can:

* Read the YAML top to bottom
* Modify steps safely during the workshop
* See how changes affect CI behavior

Advanced features like Docker builds, automated versioning, and releases are **out of scope for this workshop** and live in the full example repository instead.

---

## What You Should Focus On During the Workshop

You are **not expected** to:

* Understand every line of Python
* Memorize GitHub Actions syntax
* Optimize the ML model

You **should focus on**:

* How workflows are triggered
* How jobs and steps are structured
* How GitHub Actions uses `${{ }}` expressions
* How CI provides fast feedback on code changes

---

## Repository Structure (Minimal)

```
sst/
├── src/sst/      # Python package
├── tests/        # Test suite
├── data/         # Small sample datasets
├── artifacts/    # Generated outputs (local / CI)
├── .github/      # GitHub Actions workflows
└── pyproject.toml
```

---

## After the Workshop

If you want to see how these ideas scale to a more realistic scientific project, explore:

[https://github.com/chicago-aiscience/workshop-sst](https://github.com/chicago-aiscience/workshop-sst)

That repository applies the same CI/CD patterns you learn here, but with more automation and real-world structure.

---

## Questions

If something feels unclear or overly complex, that’s valuable workshop feedback. Please open an issue or bring it up during the session so we can improve the learning experi
