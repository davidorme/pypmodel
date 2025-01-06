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

# Developing `pyrealm`

This page gives an overview of the process of contributing code to the `pyrealm`
package, along with the development environment and tools you will need to setup to work
with the codebase.

## What is a package contributor?

Being a contributor is all about helping improve the `pyrealm` package. That could be
something very small, like fixing typos in the package website, or something large, like
adding a draft of an entirely new science module to the package.

We welcome _all_ contributions, but we need to manage contributions of code and
documentation to make sure everything works properly together and to keep the code and
documentation consistent. We do a lot of this by using some automated tools that help
keep the package well organised and ensure that it keeps giving the same results through
time.

These tools take a bit of getting used to and the rest of this document sets out how to
get your computer set up to run them. It is a good idea to start off with a small
contribution in order to get used to the workflow - please do reach out to other
developers for help in getting things to work if you run into problems. We will expect
you to have read this document and the linked details pages, but we do not expect them
to be a perfect or complete explanation!

## Contributing code

The workflow for contributing to `pyrealm` currently follows the Gitflow strategy. The
basic workflow is described below but [this AWS
link](https://docs.aws.amazon.com/prescriptive-guidance/latest/choosing-git-branch-approach/gitflow-branching-strategy.html)
provides an overview of the strategy.

1. Decide what you want to work on. This could be an existing bug or feature request or
   could be something new. If it is new, then create a new issue on Github describing
   what you want to change or add. The issue tracker provides templates for bugs and
   feature requests: please do provide as much detail as possible on the bug or the
   feature you would like to provide. If you want to work on an existing issue, then
   just add a comment and say you would like to work on it.

   [https://github.com/ImperialCollegeLondon/pyrealm/issues](https://github.com/ImperialCollegeLondon/pyrealm/issues)

   Whatever issue you do want to work on, do give other developers a chance to comment
   on suggestions before putting a lot of effort in!

1. On Github issue pages, there is a development link to "create a branch" for the
   issue. The branch name will then start with the issue number, which makes branches
   much easier to track, and is explicitly linked to the issue. Feel free to shorten the
   branch name - it uses the issue title by default.

1. Check that branch out locally and make commits to it, pushing them to GitHub
   regularly. Do try and make frequent small commits with clear, specific commit
   messages: a commit does not mean that an issue is completed, just that you want to
   record your progress. The commit history can always be compressed at the merge stage
   (see below).

1. Create a pull request (PR) from the issue branch onto the `develop` branch. The PR
   description should tag the issue being addressed and explain how the incoming code
   fixes the issue. You can start a PR as 'draft' PR: this can be a useful way to start
   describing a PR content and checking that testing is passing before opening a PR up
   for review.

   We prefer pull requests to be small, with the aim of reviewing and merging frequently
   the smallest functional unit of work that you can. This helps stop pull requests
   getting stalled on more and more complex tasks and makes code review fast.

1. Check that the continuous integration testing passes and fix any issues causing test
   failures.

1. Request reviews from other package developers using the Review section on the PR
   page. A PR cannot be merged into `develop` until at least one approving review has
   been added to the code. Reviews will often suggest changes to the code and you should
   discuss those suggestions and implement them.

   Hopefully, you will have talked to other developers during the process of writing the
   PR and should have some ideas of who to ask for a review. If not, please request
   [`davidorme`](https://github.com/davidorme) to review the PR and we can then work out
   which of the core team is best placed to give feedback.

1. Once a PR has been approved, the PR can be merged into `develop` and the branch can
   be deleted.

   The `Merge Pull Request` button provides alternative merge strategies. The default is
   to create a "merge commit" - all of the commits on the PR are merged individually to
   `develop` - but you can also "squash and commit" - which squashes all of the commits
   into a single commit and message before merging. Squashing commits can be really
   helpful to avoid a bunch of minor 'typo' commit messages, but can also make it harder
   to find commits that made bigger changes on a branch. In general, we use "merge
   commits", but if the commit history on a branch is mostly a sequence of minor edits,
   feel free to squash.

## The package development environment

The short descriptions below provide the key commands needed to set up your development
environment and provide links to more detailed descriptions of code development for
`pyrealm`. The [example setup script](#setup-script-example) below gathers
the commands together into a single script, currently only for Linux.

### Python environment

You will need to install Python to develop `pyrealm`. The package is currently tested
against the following Python versions: 3.10 and 3.11. You should install one of these
versions for developing `pyrealm`.

We highly recommend using [`pyenv`](https://github.com/pyenv/pyenv) or
[`pyenv-win`](https://github.com/pyenv-win/pyenv-win)  to manage your Python
installations. These tools allow you to manage multiple different python versions in
parallel and to switch between them. However, these extra steps are not necessary to get
started.

### Package management

We use [`poetry`](https://python-poetry.org/docs/#installation) for dependency
management and for managing development environments and you will need to install it.
The `pyrealm` package currently uses `poetry` version 1.8.2 and you should specify this
when installing to avoid conflicts with the package management process.

For the typical installation process, this would be as simple as:

```sh
curl -SSL https://install.python-poetry.org | python3 - --version 1.8.2
```

### Installing `pyrealm`

To develop `pyrealm`, you will also need to install [`git`](https://git-scm.com/) and
then clone the `pyrealm` GitHub repository.

```sh
git clone https://github.com/ImperialCollegeLondon/pyrealm.git
```

You can now use `poetry` to install the package dependencies. This is not just the
package requirements for end users of the package, but also a wider set of tools used in
package development. `poetry` uses the
[pyproject.toml](https://github.com/ImperialCollegeLondon/pyrealm/blob/develop/pyproject.toml)
file to configure the dependencies that will be installed.

```bash
poetry install
```

Poetry uses a virtual environment for package development: all packages are installed to
a stand-alone python environment that is only used for `pyrealm` development. This makes
sure that the development environment is consistent across python versions and different
developers. However, when you are working on the command line, you need to **explicitly
use the `pyrealm` environment** to run any command that needs to use the `pyrealm`
environment - and that is pretty much everything described in this document. There are
two options to do this:

1. You can add `poetry run` before a command to make sure that single command is run
   using the `poetry` environment. This approach is used in the example commands below.
1. You can use `poetry shell` to start a new shell that uses this environment: you can
   then run commands without needing `poetry run` and they should use the correct
   enviroment. This is usually more convenient.

You should now be able to run the following command to see that `pyrealm` is installed
and is showing the current version.

```sh
poetry run python -c "import pyrealm; print(pyrealm.__version__)"
```

### Updating `poetry` and package versions

You will not need to do this when setting up your development environment but one of the
things that `poetry` does is to maintain a fixed set of compatible required packages.
The `pyproject.toml` files sets constraints on package versions, but the particular
combination to be used for a given commit is resolved and stored in the `poetry.lock`
file.

* If you want to **add a package** - either using `poetry add` or by manually updating
  `pyproject.toml` - you will then need to run `poetry update` to check that a
  compatible set of package versions exists and to update the `poetry.lock` file.

* If you want to **update a package** then `poetry update` will update all the required
  packages and update `poetry.lock`. You can use `poetry update package_name` to only
  update a particular requirement.

* The `poetry install` command - as shown above - can be re-run to re-install the
  package. You will typically only need to do this if commands provided by the package
  have changed and need to be updated.

If you pull code from GitHub that changes `pyproject.toml` and `poetry.lock`, you should
also run `poetry update` to bring your environment in line with other developers.

### Installing and using `pre-commit`

Development of the `pyrealm` package uses [`pre-commit`](https://pre-commit.com/). This
is a python tool that runs a set of checks on `git` commits and stops the commit from
completing when any of those checks fail. We use `pre-commit` to help catch a wide range
of common issues and make sure that all code pushed to the GitHub repository meets some
simple quality assurance checks and uses some common formatting standards.

There is a detailed description of the `pre-commit` output and the  configured checks
and update process on the [code quality assurance page](./code_qa_and_typing.md).
Briefly, we use `pre-commit` to catch inconsistent formatting and variable typing and to
run the widely-used `flake8` code checking suite.

The `pre-commit` tool is installed by the `poetry install` step above, so you now need
to install the `pyrealm` configuration for `pre-commit` and run the tool to set up the
environment and check it is all working.

```sh
poetry run pre-commit install
poetry run pre-commit run --all-files
```

That might take a little while to run on the first use. Once you have done this, every
`git commit` will generate similar output and your commit will fail if issues are found.

### Static typing with `mypy`

The `python` programming language does not _require_ code objects to be typed, but the
`pyrealm` package uses [type hints](https://peps.python.org/pep-0484/) to annotate code.
Those type hints are then checked using the `mypy` static type checker, which is
installed by `poetry` and is run as one of the `pre-commit` checks.

The `mypy` package and the plugins we use are all installed by `poetry`. See the [code
quality assurance page](./code_qa_and_typing.md) for more information on using `mypy`.

### Package testing

All code in the `pyrealm` package should have accompanying unit tests, using `pytest`.
Look at the existing test suite in the `tests/unit` directory to see the structure and
get a feel for what they should do, but essentially unit tests should provide a set of
known inputs to a function and check that the expected answer (which could be an
Exception) is generated.

Again, the `pytest` package and plugins are installed by `poetry`. See the [code testing
page](./code_testing.md) for more details but you should be able to check the tests run
using the following command. Be warned that the `mypy` steps can be very time consuming
on the first run, but `pytest` does some cacheing that makes them quicker when they next
run.

```sh
poetry run pytest
```

### Code profiling

We use a specific set of `pytest` tests to carry out code profiling using
the `pytest-profile` plugin. This makes it easier to spot functions that are running
slowly within the `pyrealm` code. We also use our own `profiling/run-benchmarking.py`
script to compare code profiles between versions to check that new code has not impacted
performance.

See the [profiling and benchmarking page](./profiling_and_benchmarking.md) for more
details on running profiling.

### The `pyrealm-build-data` package

The `pyrealm` repository includes the [`pyrealm-build-data`
package](./pyrealm_build_data.md), which is used to provide a range of datasets used in
package testing and in documentation.

### Documentation

We use `sphinx` to maintain the documentation for `pyrealm` and Google style docstrings
using the `napoleon` formatting to provide API documentation for the code. We use MyST
Markdown to provide dynamically built usage examples. See the [documentation
page](./documentation.md) for details but to get started, the following code can be used
to build the documentation.

```bash
# Build docs using sphinx
cd docs
poetry run sphinx-build -W --keep-going source build
```

Once that command completes, the file `docs/build/html/index.html` can be opened to view
the built documentation.

### GitHub Actions

We use GitHub Action workflows to update `pre-commit`, run code quality checks on pull
requests, and to automate profiling and release publication. See the [GitHub Actions
page](./github_actions.md) for details.

### Package version releases

We use trusted publishing from GitHub releases to release new versions of `pyrealm` to
[PyPI](https://pypi.org/project/pyrealm/). Releases are also picked up and archived on
[Zenodo](https://doi.org/10.5281/zenodo.8366847). See the [release process
page](./release_process.md) for details.

## Setup script example

The scripts below bundle all the commands together to show the set up process, including
using `pyenv` to mangage `python` versions, ending by running the unit tests. This sets
up everything you need, ready to start developing on pyrealm.

:::{admonition} Setup script

``` sh
!/bin/bash

# pyenv and poetry use sqlite3. You _may_ need to install these requirements first.
sudo apt install sqlite3 sqlite3-doc libsqlite3-dev

# install pyenv to manage parallel python environments
curl <https://pyenv.run> | bash

# Manually edit .bash_profile or .profile to setup pyenv:

# export PYENV_ROOT="$HOME/.pyenv":
# [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH":
# eval "$(pyenv init -)"

# Install a python version
pyenv install 3.11

# Install poetry
curl -sSL https://install.python-poetry.org | python3 -

# Manually add poetry to path in profile file:

# export PATH="/home/validate/.local/bin:$PATH"

# Clone the repository
git clone https://github.com/ImperialCollegeLondon/pyrealm.git

# Configure the pyrealm repo to use python 3.11
cd pyrealm
pyenv local 3.11
poetry env use 3.11

# Install the package with poetry
poetry install

# Install pre-commit and check
poetry run pre-commit install
poetry run pre-commit run --all-files

# Run the test suite
poetry run pytest

```

:::
