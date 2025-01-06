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

# Package release process

The package release process has four stages. The last two steps are automated:

* Merge changes from the `develop` branch onto `main` that will form the new release
  version.
* Publish a release on GitHub - this is basically just a specific tagged commit on
  `main` that has some associated release notes.
* Publish the code on Zenodo.
* Publish the built code packages to PyPI - this is the packaged version of the code
  that users will install and use. The `pyrealm` package uses the trusted publishing
  mechanism to make it easy to add new release to PyPI.

## Generate the code commit to be released

The release process for new versions of the `pyrealm` package is managed using
pull requests to the `main` branch to create a specific commit that will be released.
The steps of the process are:

1. **Create a new release branch** from the `develop` branch called `release/X.Y.Z` ,
   where `X.Y.Z` is the expected release version number.

1. Update the `pyproject.toml` file to use the expected release versions number and
   commit that change. You can use `poetry version` command to increment the major,
   minor and patch version but it is almost as easy to edit the file by hand.

1. That commit should set the standard `pyrealm_ci.yaml` actions going, which includes
   code QA, testing and docs building. However, you should also check the documentation
   builds on Read The Docs.

   Log in to [https://readthedocs.org](https://readthedocs.org) which is the admin site
   controlling the build process. From the Versions tab, activate the `release/X.Y.Z`
   branch and wait for it to build. Check the Builds tab to see that it has built
   successfully! If it has built succesfully, do check pages to make sure that page code
   has executed successfully, and then go back to the Versions tab and deactivate and
   hide the branch. If the release branch needs any changes, do come back and check that
   those changes have also built successfully.

1. **Start a pull request against the `main` branch**. The PR will transfer all of the
   changes to the `develop` branch since the last release on to the `main` branch. The
   PR description should provide a good explanation of the functionality that is being
   changed or added in this version, and an explanation of the suggested version number
   increment. For example, "This PR fixes a bug in calculating plant growth and so is a
   patch release from v.0.1.8 to v0.1.9".

1. **The CI testing obviously now needs to pass**. Any issues need to be resolved by
   commits or PRs onto the `release/x.y.x` branch.

1. **The PR also must be reviewed**. The code itself has already gone through the
   review process to be merged into `develop`, so this is not a code review so much as a
   review of the justification for a release.

1. **The branch can then be merged into `main`**. Do _not_ delete the release branch at
   this point.

1. **Create a second PR to merge the release branch into `develop`.** This is to
   synchronise any release changes (including the version number change) between the
   `main` and `develop` branches.

## Create the GitHub release

The head of the `main` branch is now at the commit that will be released as version
`X.Y.Z`. The starting point is to **go to the [draft new release
page](https://github.com/ImperialCollegeLondon/pyrealm/releases/new)**. The
creation of a new release is basically attaching notes and files to a specific commit on
a target branch. The steps are:

1. On that release page, the **release target** dropdown should essentially always be
   set to `main`: the whole point in this branch is to act as a release branch.

1. You need to provide a tag for the commit to be released - so you need to **tag the
   commit on the `main` branch** using the format `vX.Y.Z`. You can:

   * Create the tag locally using `git tag vX.Y.Z` and then push the tag using `git push
     --tags`. You can then select the existing tag from the drop down on the release
     page.
   * Alternatively, you can simply type the tag name into that drop down and the tag
     will be created alongside the draft release.

1. You will need to choose a title for the release: basically `Release vX.Y.Z` is fine.
   However, the title text also provides a mechanism for suppressing automatic trusted
   publication to the main PyPI server by using `Release vX.Y.Z test-pypi-only`. See
   below for details.

1. You can create release notes automatically - this is basically a list of the commits
   being added since the last release - and can also set the version as a pre-release.
   This is different from having an explicit release version number (e.g. `X.Y.Za1`) -
   it is just a marker used on GitHub.

   At this point, you can either save the draft or simply publish it. It is probably
   good practice to save the draft and then have a discussion with the other developers
   about whether to publish it.

1. Once everyone is agreed **publish the release**.

## Publish the package on Zenodo

The `pyrealm` package is set up to publish on Zenodo automatically when a release is
published. This saves a zip file of the repository as a new Zenodo record, under the
[`pyrealm` concept ID](https://zenodo.org/doi/10.5281/zenodo.8366847).

This process can be turned off for test releases but only through the owning Zenodo
account, which belongs to David Orme.

## Publish the package on PyPI

We publish to _two_ package servers:

* The
  [TestPyPI](https://test.pypi.org/project/pyrealm/) server is a final check
  to make sure that the package build and publication process is working as expected.
* The package builds are then published to the main
  [PyPI](https://pypi.org/project/pyrealm/) server for public use.

The GitHub action automates the publication process but the process can also be carried
out manually.

### Manual publication

The publication process _can_ be carried out from the command line. The manual process
looks like this:

```sh
# Use poetry to create package builds in the dist directory
poetry build
# Building pyrealm (x.y.z)
#  - Building sdist
#  - Built pyrealm_x.y.z.tar.gz
#  - Building wheel
#  - Built pyrealm_x.y.z-py3-none-any.whl

# Use twine to validate publication to TestPyPI
twine upload --repository testpypi --config-file .pypirc dist/*

# Use twine to publish to PyPI
twine upload --repository pypi --config-file .pypirc dist/*
```

The tricky bit is that you need to provide a config file containining authentication
tokens to permit publication. Those tokens **must not be included in the repository**
and so need to be carefully shared with developers who make releases. If this is being
used the `.pypirc` file contents will look something like:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-longAuthToken


[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-anotherLongAuthToken
```

At this point, you should also upload the built package wheel and source files to the
assets section of the GitHub release.

### Trusted publishing

The `pyrealm` repository is set up to use trusted publishing through a Github
Actions workflow. The workflow details are shown below, along with comments, but the
basic flow is:

1. When a GitHub release is published, the PyPI publication workflow is triggered.
1. The standard continuous integration tests are run again, just to be sure!
1. If the tests pass, the package is built and the wheel and source code are stored as
   job artefacts.
1. The built files are automatically added to the release assets.
1. The job artefacts are published to the Test PyPI server, which is configured to
   automatically trust publications from this GitHub repository.
1. As long as all the steps above succeed, the job artefacts are now published to the
   main PyPI site, which is also configured to trust publications from the repository.

   The last step of publication to the main PyPI site can be skipped by including the
   text `test-pypi-only` in the title text for the release. This allows pre-release
   tests and experimentation to be tested without automatically adding them to the
   official released versions.

```{eval-rst}
.. include::  ../../../.github/workflows/pyrealm_publish.yaml
    :code: yaml
```
