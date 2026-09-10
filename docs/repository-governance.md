# Repository governance

This project uses least-privilege repository governance while preserving normal
open-source contribution workflows and the complete project history.

## Contributor model

Contributors may create and update feature branches, make commits, open pull
requests, participate in reviews and discussions, manage ordinary issues as
permitted by their role, and use the project's normal development workflows.

## Protected authority

The repository owner, `@waqasm86`, retains repository administration, final
authorization for changes to `main`, release-tag authority, PyPI deployment
approval, and authority over security-sensitive repository settings.

Ordinary contributor access does not confer permission to administer repository
roles, rulesets, protected refs, environments, or other governance controls.

## Pull-request policy

Changes to `main` are made through pull requests. Contributor-authored pull
requests must pass the required CI checks, resolve review conversations, and
receive the required CODEOWNER approval from `@waqasm86` before merge. New
reviewable commits dismiss stale approvals.

Because a pull-request author cannot approve their own pull request, an
owner-authored pull request follows the same CI requirements and may then use
the explicitly authorized owner bypass for the review requirement. The bypass
is an owner-only governance mechanism and does not grant ordinary contributors
authority to bypass protected-main policy.

## Release and publication policy

Release-style `v*` tags are owner-controlled and protected against ordinary
contributor creation, update, force-update, and deletion. Publication to PyPI
uses the `pypi` GitHub environment and requires approval from `@waqasm86` while
preserving the existing trusted-publishing OIDC workflow.

## History policy

Historical branches, tags, GitHub Releases, evidence, commits, and merged pull
requests are intentionally preserved. Governance changes must not delete or
rewrite project history.
