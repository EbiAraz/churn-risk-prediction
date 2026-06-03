# Contributing

Thank you for contributing to this project.

## Development Setup

1. Clone the repository.
2. Create a Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Typical Workflow

1. Create a branch:

```bash
git checkout -b feature/your-change
```

2. Make your changes.
3. Run a quick syntax check:

```bash
python -m py_compile "churn prediction.py" "threshold_dashboard_app.py"
```

4. Commit your changes:

```bash
git add .
git commit -m "Describe your change"
```

5. Push branch and open a pull request.

## Pull Request Checklist

- Code is readable and follows existing style.
- README is updated when behavior changes.
- New outputs/artifacts are documented.
- No unrelated files are modified.
