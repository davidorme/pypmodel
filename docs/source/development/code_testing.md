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

# Package testing and profiling

The `pyrealm` package uses `pytest` to provide benchmark tests, unit tests and
integration testing. In addition, `doctest` is used to maintain examples of code usage
in the package docstrings and ensure that the documented return values are correct.

## Using `pytest`

The `tests` directory contains modules providing test suites for each of the different
package modules. This includes:

* regression testing the output of `pyrealm` code against previously existing
  implementations of some functionality, such as the `rpmodel` and `SPLASH` packages.
* unit testing of individual functions and methods and,
* profiling and integration testing using combinations of modules.

These are the main tests that ensure that the package is behaving as expected and that
it produces stable outputs. The test suite can be run from repository using:

```bash
poetry run pytest
```

The `setup.cfg` file contains `pytest` configuration details. By default, we do not
include the slow-running `profiling` tests as part of the standard `pytest` suite. See
the [profiling page](./profiling_and_benchmarking.md) for details on running those
tests.

## Using `doctest`

Some of the package docstrings contain `doctest` examples of code use. These examples
are intended to provide simple examples of method or function use and generate an
output: the `doctest` module is used to make sure that the code runs and gives the
expected result.

We have configured `pytest` to automatically also run `doctest`, but you can manually
check the tests in files using, for example:

```bash
poetry run python -m doctest pyrealm/pmodel/pmodel.py
```

Normally, `doctest` is just used to test a return value: the value tested is the value
printed to the console, so it is common to use some form of `round` to make sure values
match. It can also be used to check that an error or warning is raised. See the
docstring for {meth}`~pyrealm.core.hygro.convert_rh_to_vpd` to see how checking for
warning text can be included in a doctest.

## Using `pytest-coverage` and `codecov`

Using the plugin [pytest-coverage](https://pypi.org/project/pytest-cov/) you can
generate coverage reports. You can run:

```bash
poetry run pytest --cov=<test_path>
```

to perform coverage analysis. The report is stored with the name `index.html`. It can be
used to determine if your contribution is adequately tested. The GitHub Actions
[continuous integration workflow](./github_actions.md#pyrealm_ciyaml) automatically
uploads coverage data to the
[CodeCov](https://app.codecov.io/gh/ImperialCollegeLondon/pyrealm) website.
