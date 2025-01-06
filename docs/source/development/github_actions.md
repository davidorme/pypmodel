---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
---

# GitHub Actions

The project uses several workflows using GitHub Actions to maintain code quality and
confirm that the package and website are building correctly. The actions are defined in
`.github/workflows`directory and currently include:

## `pre-commit_autoupdate.yaml`

This workflow runs every week at midnight on Monday and creates a new pull request to
update the `pre-commit` actions.

## `pyrealm_ci.yaml`

This workflow runs when a pull request is opened and when new commits are made to an
existing pull request. It is the main quality assurance check on new code and runs three
jobs:

* code quality assurance (`qa`): does the code pass all the `pre-commit` checks.
* code testing (`test`): do all the unit and regression tests pass.
* documentation building (`docs_build`): does the documentation build correctly.

If any of those checks fail, you will need to push new commits to the pull request to
fix the outstanding issues. The status of code checking for pushed commits can be seen at:

[https://github.com/ImperialCollegeLondon/pyrealm/actions](https://github.com/ImperialCollegeLondon/pyrealm/actions)

Although GitHub Actions automates these steps for any pushes, pull requests and releases
on the repository, you should also perform the same steps locally before submitting code
to ensure that your code passes testing.

## `pyrealm_publish.yaml`

This workflow runs when a release is made on the GitHub site and uses trusted publishing
to build the package and publish it on [PyPI](https://pypi.org/project/pyrealm/).
