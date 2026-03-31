# Contributing

Thanks for contributing to `hiveframe`.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run tests

```bash
pytest
```

## Coding guidelines

- Keep changes focused and production-ready.
- Prefer clean, maintainable code over long documentation.
- Add or update tests when behavior changes.
- Keep public APIs backward compatible unless intentionally changed.

## Pull request checklist

- [ ] Tests pass locally.
- [ ] New behavior is covered by tests.
- [ ] README/examples are updated if user-facing behavior changes.
- [ ] Commit messages are clear and scoped.

## AI-assisted contributions

We welcome AI-assisted development. However:

### MUST
- [ ] You understand every line you submit
- [ ] You can explain any change in plain language
- [ ] All tests pass: `pytest -v`
- [ ] No new dependencies without prior discussion
- [ ] Changes are backward compatible unless 
      explicitly discussed in issue first

### MUST NOT
- [ ] Do not add abstraction layers not requested
- [ ] Do not change public API without RFC process
- [ ] Do not add features outside issue scope
- [ ] Do not submit AI output without review

### How we detect vibe coding without understanding
We will ask you to explain your changes in PR review.
If you cannot explain why a specific decision was made,
the PR will be closed regardless of test results.