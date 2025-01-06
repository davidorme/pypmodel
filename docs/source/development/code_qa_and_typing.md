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

# Code quality and static typing

We use `pre-commit` to ensure common code standards and style, and use `mypy` to provide
static typing of the `pyrealm` codebase.

## Using `pre-commit`

As described in the [developer overview](./overview.md), `pre-commit` is installed as
part of the `pyrealm` developer dependencies and so can be set up to run simply using:

```sh
poetry run pre-commit install
poetry run pre-commit run --all-files
```

This can take a while on the first run, and when the configuration updates, as the tool
needs to install or update all the hooks that are applied to changes within a commit.
Usually the hooks only run on files changed by a particular `git commit` but using
`pre-commit run --all-files` scans the entire codebase and is a commonly used check to
make sure all is well.

### The `pre-commit` configuration

The file
[.pre-commit-config.yaml](https://github.com/ImperialCollegeLondon/pyrealm/blob/develop/.pre-commit-config.yaml)
contains the pre-commit hooks used by `pyrealm`. The configuration file contains links
to each individual hook but in overview:

`check for merge conflicts`
: Checks for remaning `git` merge conflict markers in code files.

`debug statements (python)`
: Checks for debugger imports and `breakpoint()` calls, which should not end up in
released code.

`pyupgrade`
: Updates Python syntax to the current Python 3.10 syntax.

`isort`
: Enforces a consistent sort order and formatting of package imports in modules.

`black`
: Enforces a common code formatting across the codebase. This can be irritating but it
keeps code neatly formatted and avoids code changes that are simply alternate
formatting.

`flake8`
: This runs the [`flake8](https://flake8.pycqa.org/en/latest/) tool to detect a very
wide range of common programming issues. We use the default set of plugins to check: [PEP
8 style
recommendations](https://pycodestyle.pycqa.org/en/latest/intro.html#error-codes), [code
complexity](https://pypi.org/project/mccabe/) and [common programming
errors](https://flake8.pycqa.org/en/latest/user/error-codes.html). It is also configured
to check docstrings on code objects using
[`pydocstyle`](https://www.pydocstyle.org/en/stable/error_codes.html) via the
`flake8-docstrings` plugin.

`mypy`
: Runs static typing checks to ensure that the types of function arguments and return
values are declared and are compatible. See [below](#typing-with-mypy) for more
information.

`markdownlint`
: Checks all markdown files for common formatting issues.

### Output and configuration

When `pre-commit` runs, you may see some lines about package installation and update,
but the key information is the output below, which shows the status of the checks set up
by each hook:

```text
check for merge conflicts................................................Passed
debug statements (python)................................................Passed
pyupgrade................................................................Passed
isort....................................................................Passed
black....................................................................Passed
flake8...................................................................Passed
mypy.....................................................................Passed
markdownlint.............................................................Passed
```

### Updating `pre-commit`

The hooks used by `pre-commit` are constantly being updated to provide new features or
to update code to deal with changes in the implementation. You can update the hooks
manually using `pre-commit autoupdate`, but the configuration is regularly updated
through our GitHub Actions workflows

## Typing with `mypy`

Unlike many programming languages, Python does not require variables to be declared as
being of a particular type. For example, in C++, this code creates a variable that is
_explicitly_ an integer and a function that _explicitly_ requires an integer and returns
an integer value. This is called **typing**.

```c++
int my_integer = 15;

int fun(int num) {

  printf("num = %d \n", num);

  return 0;
}
```

Python does not require explicit typing. That can be very useful but it can also make it
very difficult to be clear what kinds of variables are being used. The `pyrealm` project
requires static typing of the source code: the syntax for this started with [PEP
484](https://peps.python.org/pep-0484/) and a set of quality assurance tools have
developed to help support clear and consistent typing. We use
[`mypy`](https://mypy.readthedocs.io/en/stable/) to check static typing. It does take a
bit of getting used to but is a key tool in maintaining clear code and variable
structures.

## Supressing checking

The `pre-commit` tools sometimes complain about things that we do not want to change.
Almost all of the tools can be told to suppress checking, using comments with a set
format to tell the tool what to do.

This should not be done lightly: we are using these QA tools for a reason.

* `isort` allows various `# isort: skip` [action comments](https://pycqa.github.io/isort/docs/configuration/action_comments.html)
* `black` allows lines or section to be [left
  untouched](https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#ignoring-sections)
  using, for example `# fmt: skip`.
* `flake8` uses the `# noqa` comment to [suppress
  warnings](https://flake8.pycqa.org/en/3.0.1/user/ignoring-errors.html#in-line-ignoring-errors).
  For `pyrealm` you should  explicitly list the errors to be ignored, so that other
  errors are not missed: `# noqa D210, D415`.
* `mypy` uses the syntax `# type: ignore` comment to [suppress
  warnings](https://mypy.readthedocs.io/en/stable/error_codes.html#silencing-errors-based-on-error-codes).
  Again, `pyrealm` requires that you provide the specific `mypy` error code to be
  ignored to avoid missing other issues:  `# type: ignore[operator]`.
* `markdownlint` catches issues in Markdown files and uses a range of [HTML comment
  tags](https://github.com/DavidAnson/markdownlint?tab=readme-ov-file#configuration) to
  suppress format warnings. An example is `<!-- markdownlint-disable-line MD001 -->` and
  a list of the rule codes can be found
  [here](https://github.com/DavidAnson/markdownlint/blob/main/doc/Rules.md).
