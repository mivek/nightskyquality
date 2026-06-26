# Contributing

## Commit Messages

This repo uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>: <short description>

[optional body]
```

Types:
- `feat` — new feature
- `fix` — bug fix
- `docs` — documentation
- `style` — formatting (no logic change)
- `refactor` — code restructuring
- `perf` — performance improvement
- `test` — adding/fixing tests
- `chore` — maintenance, tooling, dependencies

Breaking changes: append `!` after the type, e.g. `feat!: drop Python 3.6 support`.

## Pull Requests

PR titles must also follow conventional commits. PRs with non-conforming commit messages will be flagged by CI.
