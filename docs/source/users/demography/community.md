---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
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

# Plant Communities

:::{admonition} Warning

This area of `pyrealm` is in active development.

:::

```{code-cell} ipython3
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyrealm.demography.flora import PlantFunctionalType, Flora
from pyrealm.demography.community import Cohorts, Community
```

```{code-cell} ipython3
short_pft = PlantFunctionalType(
    name="short", h_max=15, m=1.5, n=1.5, f_g=0, ca_ratio=380
)
tall_pft = PlantFunctionalType(name="tall", h_max=30, m=1.5, n=2, f_g=0.2, ca_ratio=500)

# Create the flora
flora = Flora([short_pft, tall_pft])

# Create a simply community with three cohorts
# - 15 saplings of the short PFT
# - 5 larger stems of the short PFT
# - 2 large stems of tall PFT

community = Community(
    flora=flora,
    cell_area=32,
    cell_id=1,
    cohorts=Cohorts(
        dbh_values=np.array([0.02, 0.20, 0.5]),
        n_individuals=np.array([15, 5, 2]),
        pft_names=np.array(["short", "short", "tall"]),
    ),
)
```

```{code-cell} ipython3
community
```
