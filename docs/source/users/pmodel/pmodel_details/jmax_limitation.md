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

# $J_{max}$ limitation

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import pyplot
import numpy as np
from pyrealm.pmodel import PModel, PModelEnvironment

%matplotlib inline

# Set the resolution of examples
n_pts = 201

# Create a range of representative values for key inputs.
tc_1d = np.linspace(-25, 50, n_pts)
```

Environmental conditions can also lead to limitation of both the electron transfer rate
and the carboxylation capacity ($V_{cmax}$) of leaves. The
{class}`~pyrealm.pmodel.pmodel.PModel` implements three alternative approaches to the
calculation of $J_{max}$ and $V_{cmax}$, using the argument `method_jmaxlim`. These
options set the calculation of two factor ($f_j$ and $f_v$) which are applied to the
calculation of $J_{max}$ and $V_{cmax}$. The options for this setting are:

* `simple`: This approach implements the 'simple' formulations of the P Model, with no
  limitations and hence $f_j  = f_v = 1$.
* `wang17`: This is the default setting for `method_jmaxlim` and applies the
  calculations describe in  {cite:t}`Wang:2017go`. The calculation details can be
  seen in the {meth}`~pyrealm.pmodel.jmax_limitation.JmaxLimitation.wang17` method.

* `smith19`: This is an alternate calculation for optimal values of $J_{max}$
  and $V_{cmax}$ described in {cite:t}`Smith:2019dv`. The calculation details can be
  seen in the {meth}`~pyrealm.pmodel.jmax_limitation.JmaxLimitation.smith19` method.

The plot below shows the effects of each method on the light use efficienct across a
temperature gradient. The other forcing variables are fixed ($P=101325.0 , \ce{CO2}= 400
\text{ppm}, \text{VPD}=820$) and $\phi_0$ is also fixed ($\phi_0=0.08$).

```{code-cell} ipython3
:tags: [hide-input]

# Calculate variation in m_jlim with temperature
env = PModelEnvironment(tc=tc_1d, patm=101325, vpd=820, co2=400)

model_jmax_simple = PModel(
    env, method_jmaxlim="simple", method_kphio="fixed", reference_kphio=0.08
)
model_jmax_wang17 = PModel(
    env, method_jmaxlim="wang17", method_kphio="fixed", reference_kphio=0.08
)
model_jmax_smith19 = PModel(
    env, method_jmaxlim="smith19", method_kphio="fixed", reference_kphio=0.08
)

# Create a line plot of the resulting values of m_j
pyplot.plot(tc_1d, model_jmax_simple.lue, label="simple")
pyplot.plot(tc_1d, model_jmax_wang17.lue, label="wang17")
pyplot.plot(tc_1d, model_jmax_smith19.lue, label="smith19")

pyplot.title("Effects of J_max limitation")
pyplot.xlabel("Temperature °C")
pyplot.ylabel("Light Use Efficiency (g C mol-1)")
pyplot.legend()
pyplot.show()
```
