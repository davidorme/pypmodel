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

# Soil moisture effects

At present, there are four approaches for incorporating soil moisture effects on
photosynthesis:

* Two soil moisture stress functions, both of which estimate a penalty factor to
  calculated GPP based on soil moisture conditions and aridity.
  * {func}`~pyrealm.pmodel.functions.calc_soilmstress_stocker` calculates
    $\beta(\theta)$ {cite:p}`Stocker:2020dh`.
  * {func}`~pyrealm.pmodel.functions.calc_soilmstress_mengoli` calculates
  * $\beta(\theta)$ {cite:p}`mengoli:2023a`.
* The experimental `rootzonestress` methods for
  {class}`~pyrealm.pmodel.optimal_chi.OptimalChiABC`, which impose a direct stress
  penalty on $\beta$.
* The `lavergne20_c3` and `lavergne20_c4` methods for
  {class}`~pyrealm.pmodel.optimal_chi.OptimalChiABC`, which use an empirical model
  of the
  change in the ratio of the photosynthetic costs of carboxilation and transpiration.
  Altering this cost ratio - inconveniently also called $\beta$ - for soil moisture
  stress provides a more complete picture of plant responses than GPP penalty factors.

The GPP penalty functions are described here, but see the [optimal chi](./optimal_chi)
documentation for details of the other methods.

## The {func}`~pyrealm.pmodel.functions.calc_soilmstress_stocker` penalty factor

This is an empirically derived factor ($\beta(\theta) \in [0,1]$,
{cite:p}`Stocker:2018be,Stocker:2020dh` that describes a penalty to gross primary
productivity (GPP)  resulting from soil moisture stress.

The factor requires estimates of:

* relative soil moisture ($m_s$, `soilm`), as the fraction of field capacity, and
* a measure of local mean aridity ($\bar{\alpha}$, `meanalpha`), as the average annual
  ratio of AET to PET.

```{admonition} Soil moisture
The parameters used in the calculation of this factor were estimated using the
plant-available soil water expressed as a fraction of available water holding capacity.
That capacity was calculated for the observed data on a site by site basis using the
`SoilGrids` dataset {cite:p}`hengl:2017a`. Ideally, soil moisture calculated in the same
way should be used with this approach.
```

The functions to calculate $\beta(\theta)$ are based on four parameters, derived from
experimental data and set in {class}`~pyrealm.constants.pmodel_const.PModelConst`:

* An upper bound in relative soil moisture ($\theta^\ast$, `soilmstress_thetastar`),
  above which $\beta$ is always 1, corresponding to no loss of light use efficiency.
* An lower bound in relative soil moisture ($\theta_0$, `soilmstress_theta0`),
  below which LUE is always zero.
* An intercept (a, `soilmstress_a`) for the aridity sensitivity parameter $q$.
* A slope (b, `soilmstress_b`) for the aridity sensitivity parameter $q$.

The aridity measure ($\bar{\alpha}$) is first used to set an aridity sensitivity
parameter ($q$), which sets the speed with which $\beta(\theta) \to 0$ as $m_s$
decreases.

$$
    q = (1 - (a + b \bar{\alpha}))/(\theta^\ast - \theta_{0})^2
$$

Then, relative soil moisture ($m_s$) is used to calculate the soil moisture factor:

$$
    \beta(\theta) = q ( m_s - \theta^\ast) ^ 2  + 1
$$

The figures below shows how the aridity sensitivity parameter ($q$) changes with
differing aridity measures and then how the soil moisture factor $\beta(\theta)$
varies with changing soil moisture for some different values of mean aridity. In
the examples below, the default $\theta_0 = 0$ has been changed to $\theta_0 =
0.1$ to make the lower bound more obvious.

```{code-cell} ipython3
:tags: [hide-input]

from matplotlib import pyplot as plt
import numpy as np
from pyrealm import pmodel
from pyrealm.pmodel.pmodel import PModel
from pyrealm.pmodel.pmodel_environment import PModelEnvironment
from pyrealm.constants import PModelConst

# change default theta0 parameter
const = PModelConst(soilmstress_theta0=0.1)

# Calculate q
mean_alpha_seq = np.linspace(0, 1, 101)

q = (1 - (const.soilmstress_a + const.soilmstress_b * mean_alpha_seq)) / (
    const.soilmstress_thetastar - const.soilmstress_theta0
) ** 2

# Create a 1x2 plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot q ~ mean_alpha
ax1.plot(mean_alpha_seq, q)
ax1.set_xlabel(r"Mean aridity, $\bar{\alpha}$")
ax1.set_ylabel(r"Aridity sensitivity parameter, $q$")

# Plot beta(theta) ~ m_s for 5 values of mean alpha
soilm = np.linspace(0, 0.7, 101)

for mean_alpha in [0.9, 0.5, 0.3, 0.1, 0.0]:

    soilmstress = pmodel.calc_soilmstress_stocker(
        soilm=soilm, meanalpha=mean_alpha, pmodel_const=const
    )
    ax2.plot(soilm, soilmstress, label=r"$\bar{{\alpha}}$ = {}".format(mean_alpha))

ax2.axvline(x=const.soilmstress_thetastar, linestyle="--", color="black")
ax2.axvline(x=const.soilmstress_theta0, linestyle="--", color="black")

secax = ax2.secondary_xaxis("top")
secax.set_xticks(
    ticks=[const.soilmstress_thetastar, const.soilmstress_theta0],
    labels=[r"$\theta^\ast$", r"$\theta_0$"],
)

ax2.legend()
ax2.set_xlabel(r"Relative soil moisture, $m_s$")
ax2.set_ylabel(r"Empirical soil moisture factor, $\beta(\theta)$")


plt.show()
```

+++ {"user_expressions": []}

### Application of the {func}`~pyrealm.pmodel.functions.calc_soilmstress_stocker` factor

The factor can be applied to the P Model by using
{func}`~pyrealm.pmodel.functions.calc_soilmstress_stocker` to calculate the factor
values and then multiplying calculated GPP ({attr}`~pyrealm.pmodel.pmodel.PModel.gpp`)
by the resulting factor. The example below shows how the predicted light use
efficiency from the P Model changes across an aridity gradient both with and without the
soil moisture factor.

```{code-cell} ipython3
# Calculate the P Model in a constant environment
tc = np.array([20] * 101)
sm_gradient = np.linspace(0, 1.0, 101)

env = PModelEnvironment(tc=tc, patm=101325.0, vpd=820, co2=400)
model = PModel(env)
model.estimate_productivity(fapar=1, ppfd=1000)

# Calculate the soil moisture stress factor across a soil moisture gradient
# at differing aridities

gpp_stressed = {}

for mean_alpha in [0.9, 0.5, 0.3, 0.1, 0.0]:
    # Calculate the stress for this aridity
    sm_stress = pmodel.calc_soilmstress_stocker(
        soilm=soilm, meanalpha=mean_alpha, pmodel_const=const
    )
    # Apply the penalty factor
    gpp_stressed[mean_alpha] = model.gpp * sm_stress
```

```{code-cell} ipython3
:tags: [hide-input]

plt.plot(sm_gradient, model.gpp, label="No soil moisture penalty")

for ky, val in gpp_stressed.items():
    plt.plot(soilm, val, label=r"$\bar{{\alpha}}$ = {}".format(ky))

plt.xlabel(r"Relative soil moisture, $m_s$, -")
plt.ylabel(r"GPP")
plt.legend()
plt.show()
```

```{warning}
In the `rpmodel` implementation, the soil moisture factor is applied within the
calculation of the P Model and the penalised GPP is used to modify $V_{cmax}$ and
$J_{max}$, so that these values are congruent with the resulting penalised LUE and GPP.

This is **not** implemented in the {mod}`~pyrealm.pmodel` module. The empirical
correction is only as a post-hoc penalty to GPP, which facilitates the comparison of
different penalty factors applied to the same P Model instance.

```

## The {func}`~pyrealm.pmodel.functions.calc_soilmstress_mengoli` penalty factor

This is an empirically derived factor ($\beta(\theta) \in [0,1]$,
{cite:p}`mengoli:2023a` that describes a penalty to gross primary productivity (GPP)
resulting from soil moisture stress.

The factor requires estimates of:

* relative soil moisture ($m_s$, `soilm`), as the fraction of field capacity, and
* a climatological estimate of local aridity index, typically calculated as total PET
  over total precipitation for an appropriate period, typically at least 20 years.

```{admonition} Soil moisture

The parameters used in the calculation of this factor were estimated using the
plant-available soil water expressed as the ratio of millimeters of soil moisture over
the total soil capacity. The soil water estimated using CRU data in SPLASH v1 model
{cite:p}`davis:2017a`, which enforces a constant soil capacity of 150mm. Again, ideally,
soil moisture calculated in the same way should be used with this approach.
```

The calculation of $\beta(\theta)$ is based on two functions of the aridity index: both
power laws, constrained to take a maximum of 1 (no soil moisture stress penalty). The
first function describes the maximal attainable level ($y$) and the second function
describes a threshold ($\psi$) at which that level is reached. The parameters of these
power laws are derived from experimental data and set in
{class}`~pyrealm.constants.pmodel_const.PModelConst`.

$$
\begin{align*}
y &= \min( a  \textrm{AI} ^ {b}, 1)\\
\psi &= \min( a  \textrm{AI} ^ {b}, 1)\\
\end{align*}
$$

```{code-cell} ipython3
from pyrealm.constants import PModelConst

const = PModelConst()
aridity_index = np.arange(0.35, 7, 0.1)

y = np.minimum(
    const.soilm_mengoli_y_a * np.power(aridity_index, const.soilm_mengoli_y_b), 1
)


psi = np.minimum(
    const.soilm_mengoli_psi_a * np.power(aridity_index, const.soilm_mengoli_psi_b),
    1,
)

plt.plot(aridity_index, y, label="Maximum level ($y$)")
plt.plot(aridity_index, psi, label="Critical threshold ($\psi$)")
plt.xlabel(r"Aridity Index (AI)")
plt.ylabel(r"$\beta(\theta)$")
plt.legend()
plt.show()
```

The penalty factor $\beta(\theta)$ is then calculated given the relative soil moisture
$\theta$, the threshold $\psi$ and the maximum level:

$$
\beta(\theta) =
    \begin{cases}
        y, & \theta \ge \psi \\
        \dfrac{y}{\psi} \cdot \theta, & \theta \lt \psi
    \end{cases}
$$

```{code-cell} ipython3
# Calculate the soil moisture stress factor across a soil moisture
# gradient for different aridity index values
beta = {}
ai_vals = [0.3, 1, 3, 6]

for ai in ai_vals:
    beta[ai] = pmodel.calc_soilmstress_mengoli(
        soilm=sm_gradient, aridity_index=np.array(ai)
    )
    plt.plot(sm_gradient, beta[ai], label=f"AI = {ai}")

plt.xlabel(r"Relative soil moisture $\theta$")
plt.ylabel(r"$\beta(\theta)$")
plt.legend()
plt.show()
```

As the plot above shows, plants in arid conditions have severely constrained maximum
levels, but have lower critical thresholds, allowing them to maintain the maximum level
under drier conditions.

### Application of the {func}`~pyrealm.pmodel.functions.calc_soilmstress_mengoli` factor

As with  {func}`~pyrealm.pmodel.functions.calc_soilmstress_stocker`, the factor is first
calculated and then applied to the GPP calculated for a model
({attr}`~pyrealm.pmodel.pmodel.PModel.gpp`). In the example below, the result is
obviously just $\beta(\theta)$ from above scaled to the constant GPP.

```{code-cell} ipython3
for ai in ai_vals:

    plt.plot(sm_gradient, model.gpp * beta[ai], label=f"AI = {ai}")

plt.xlabel(r"Relative soil moisture $\theta$")
plt.ylabel("GPP")
plt.legend()
plt.show()
```

```{code-cell} ipython3

```
