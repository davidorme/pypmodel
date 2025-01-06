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

# Package constants

The models presented in this package rely on a relatively large number of underlying
constants. In order to keep the argument lists of functions and classes as simple as
possible, the `pyrealm` package discriminates between two kinds of variables.

1. **Arguments**: Class and function arguments cover the variables that a user
is likely to want to vary within a particular study, such as temperature or
primary productivity.

2. **Model Constants**: These are the underlying values that are likely to be constant
for a particular study. These may be true constants (such as the universal gas constant
$R$) but can also be experimental estimates of coefficients for functions describing
plant physiology or geometry, which a user might want to alter to update with a new
estimate or explore sensitivity to variation.

For this reason, the `pyrealm` package provides data classes that contain sets
of default model constants. The core API and details for each class can be seen
[here](../api/constants_api).

## Creating constant class instances

These can be used to generate the default set of model parameters:

```{code-cell} ipython3
from pyrealm.constants import CoreConst, TModelTraits

core_const = CoreConst()
tmodel_const = TModelTraits()

print(core_const)
print(tmodel_const)
```

And individual values can be altered using the parameter arguments:

```{code-cell} ipython3
# Estimate processes under the moon's gravity...
core_const_moon = CoreConst(k_G=1.62)
# ... allowing a much greater maximum height
tmodel_const_moon = TModelTraits(h_max=200)

print(core_const_moon.k_G)
print(tmodel_const_moon.h_max)
```

In order to ensure that a set of parameters cannot change while models are being run,
instances of these parameter classes are **frozen**. You cannot  edit an existing
instance and will need to create a new instance to use different parameters.

```{code-cell} ipython3
:tags: [raises-exception]

core_const_moon.k_G = 9.80665
```

## Exporting and reloading parameter sets

All parameter classes inherit methods from the base
{class}`~pyrealm.constants.base.ConstantsClass` class that provides bulk import and
export of parameter settings to dictionaries and to JSON formatted files. The code below
shows these methods working. First, a trait definition in a JSON file is read into a
dictionary:

```{code-cell} ipython3
import json
import pprint

trt_dict = json.load(open("../files/traits.json", "r"))
pprint.pprint(trt_dict)
```

That dictionary can  then be used to create a TModelTraits instance using
the {meth}`~pyrealm.constants.base.ConstantsClass.from_dict` method. The
{meth}`~pyrealm.constants.base.ConstantsClass.from_json` method allows this to
be done more directly and the resulting instances are identical.

```{code-cell} ipython3
traits1 = TModelTraits.from_dict(trt_dict)
traits2 = TModelTraits.from_json("../files/traits.json")

print(traits1)
print(traits2)

traits1 == traits2
```
