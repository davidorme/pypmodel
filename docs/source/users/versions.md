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

# Versions and migration

The `pyrealm` package is still being actively developed and the functionality in the
package is being extended and revised. This document is describes the main changes in
functionality between major versions and provides code migration notes for changes in
the API of `pyrealm` code.

## Migration from 1.0.0 to version 2.0.0

There are two major API changes to existing code.

1. The signatures for both the {class}`~pyrealm.pmodel.pmodel.PModel` and
   {class}`~pyrealm.pmodel.subdaily.SubdailyPModel` classes have changed the way in
   which the quantum yield parameter ($\phi_0$) is set. The classes now require a
   specific named method for setting $\phi_0$. Each method has an associated default
   reference value of $\phi_0$, but these can be overridden. This new API makes it
   possible to extend the set of approaches for calculating quantum yield, and is
   motivated by the addition of the `sandoval` method for estimating effects of soil
   moisture stress on $\phi_0$.

    This change replaces the previous `kphio` and `do_ftemp_kphio` arguments.

    ```{code-block} ipython
    # Old syntax for fixed kphio
    mod = PModel(env, kphio=0.125, do_ftemp_kphio=False)
    # New syntax
    mod = PModel(env, method_kphio='fixed', reference_kphio=0.125)


    # Old syntax for temperature variable kphio
    mod = PModel(env, kphio=0.125, do_ftemp_kphio=True)
    # New syntax
    mod = PModel(env, method_kphio='temperature', reference_kphio=0.125)
    ```

1. The {mod}`pyrealm.pmodel.functions` module used to provide `calc_ftemp_kphio` and
   `calc_ftemp_inst_vcmax`. These functions overlapped a lot but were tuned to specific
   enzyme systems. They have been depracated and replaced with the more general function
   `calc_modified_arrhenius_factor` that can be used more widely across the package.

## Version details

```{include} ../../../CHANGES.md
    :start-line: 2
```
