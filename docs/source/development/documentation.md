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

# Documentation

This page describes the documentation of the `pyrealm` package, which is hosted at:

[https://pyrealm.readthedocs.io](https://pyrealm.readthedocs.io)

Those pages include both written guides and overview pages and API documentation
generated automatically from docstrings within the code.

## Documentation guide

The `docs/source` directory contains the content and `sphinx` configuration to build the
package website. The content directories are:

* The `users` directory contains user guides and code examples.
* The `development` directory contains details on code development for `pyrealm`.
* The `api` directory contains some simple stub files that are used to link to API
  content generated from docstrings.

### MyST Markdown

All of the documentation in `docs/source` uses [MyST
Markdown](https://myst-parser.readthedocs.io/en/latest/) rather than the
reStructuredText (`.rst`) format. Markdown is easier to write and read and the MyST
Markdown extension is a literate programming format that allows Markdown pages to be run
using Jupyter to generate dynamic content to show package use.

We only include markdown format files (`.md`) in the repository and not the standard
Python Notebook (`.ipynb`) format. Although this means that the output of code cells
does not display on GitHub, the JSON format `.ipynb` files are much bulkier and also
change whenever the notebook is run, not just when there are meaningful changes to the
content.

MyST Markdown notebooks need to contain a `YAML` format header that provides details on
how to run the notebook. The content below should appear right at the top of the file.

```yaml
---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
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
```

When the package documentation is built in `sphinx` the `myst_nb` extension
automatically runs any code in MyST markdown documents and renders the resulting code
and outputs into HTML. The documentation can therefore contain dynamically generated
examples of running code and output plots to demonstrate how to use the
package. This build process also ensures that code in documentation is kept up to date
with the package. The syntax for including code blocks in documents is shown below -
this example includes a tag that causes the code to be concealed in the built
documentation, but with a toggle button to allow it to be expanded.

````{code-block}
```{code-cell} python
:tags: [hide-input]
# This is just an example code cell to demonstrate how code is included in
# the pyrealm documentation.
```
````

### Table of contents

We use the `sphinx_external_toc` package to maintain a table of contents for the
package. The file `docs/source/_toc.yml` contains the structure of the table and you
will need to add new documentation files to this file for them to appear in the table.
The documentation build process will fail if it finds files in `docs/source` that are
not included in the table of contents!

### Docstrings

The `pyrealm` package uses docstrings written in the [Google
style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
This allows the function documentation to be stored alongside the code and it is included
in the documentation using the `sphinx` `autodoc` extension. See the code itself for
examples of the documentation formatting and typical content.

At the moment, we use the `autodoc` plugins for `sphinx` to convert docstrings to HTML
and build the online API documentation. Unfortunately, the `autodoc` package is
hard-coded to expect docstrings to use reStructuredText, which means that at the moment
**all docstrings have to be written in `rst` format**. At some point, we'd like to
switch away to using Markdown throughout, but for the moment look at the existing
docstrings to get examples of how the formatting differs.

### Referencing

Both the `docs/source` and docstrings uses the `sphinxcontrib-bibtex` package to support
citations. This uses Latex like citation keys in the documentation to insert references
and build a bibliography. The `sphinx` configuration in `docs/source/conf.py` provides a
custom Author/Year citation style. The reference library in `source/refs.bib` needs to
be kept up to date with the literature for the project.

The three common use cases are shown below using a couple of reference tags
(`Prentice:2014bc` and `Wang:2017go`) that are inclued provided in the current
[reference library](../refs.bib).

* Cite with date in parentheses (``{cite:t}`Prentice:2014bc` ``): the model implemented
  in {cite:p}`Prentice:2014bc`.
* Cite with reference(s) in parentheses (``{cite:p}`Prentice:2014bc,Wang:2017go` ``):
  using the P Model {cite:t}`Prentice:2014bc,Wang:2017go`.
* Cite as above but suppressing the parentheses to allow text before or after the
  citation (``(see {cite:alp}`Prentice:2014bc` for details)``): the class implements
  the P Model (see {cite:alp}`Prentice:2014bc` for details).

## Building the documentation

The `sphinx` package is used to build an HTML version of the package documentation
provided in `docs/source` and to include the API documentation provided in the code
docstrings. The `sphinx` building process requires some extra packages, but these are
included in the `docs` group in `pyproject.toml` and should be installed.

In order to build the package documentation, Jupyter needs to be able to associate the
documentation files with the Python environment managed by `poetry`. Fortunately, the
`poetry shell` and `poetry run` commands update the Jupyter kernel specifications so
that the `python3` kernel name points to the `poetry` environment. For example:

```bash
$ poetry run jupyter kernelspec list
Available kernels:
  ...
  python3            /Users/dorme/Library/Caches/pypoetry/virtualenvs/pyrealm-QywIOHcp-py3.10/share/jupyter/kernels/python3
```

In order to build the package documentation, the following command can then be used:

```bash
# Build docs using sphinx
cd docs
poetry run sphinx-build -W --keep-going source build
```
