---
jupytext:
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
settings:
  output_matplotlib_strings: remove
---

# The canopy model

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

```{code-cell} ipython3
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from pyrealm.demography.flora import PlantFunctionalType, Flora
from pyrealm.demography.community import Cohorts, Community
from pyrealm.demography.crown import CrownProfile, get_crown_xy
from pyrealm.demography.canopy import Canopy
from pyrealm.demography.tmodel import StemAllometry
```

The `canopy` module in `pyrealm` is used to calculate a model of the light environment
across all of cohorts within a plant [community](./community.md). Each cohort consists
of:

* a number of identically-sized individuals,
* of the same [plant functional type (PFT)](./flora.md),
* that share the same [stem allometry](./t_model.md) and [crown model](./crown.md).

The purpose of the `canopy` module is to estimate how light is absorbed down through the
canopy and allow the absorption of incoming light at different heights in the canopy to
be partitioned across stems within each cohort.

## Light extinction for a single stem

The key variables in determining the light environment for a given stem are as follows:

* The projected crown area ${A}_{p}(z)$ sets how crown area accumulates, given the crown
  shape, from the top of the stem to the ground.
* The projected leaf area $\tilde{A}_{cp}(z)$ modifies the crown area to allow for the
  vertical displacement of crown area by the crown gap fraction.
* The leaf area index $L$ for the PFT is a simple factor that sets the leaf density of
  the crown, allowing stems with identical crown area to vary in the density of actual
  leaf surface for light capture. Values of $L$ are always expressed as the area of leaf
  surface per square meter.
* The extinction coefficient $k$ for a PFT sets how much light is absorbed when passing
  through the leaf surface of that PFT.

For a single stem, the fraction of light absorbed through the entire crown is described
by the Beer-Lambert law:

$$
f_{abs} = 1 - e^{-kL}
$$

However, to calculate a vertical profile of light extinction through a crown with total
area $A_c$ and maximum stem height $H$, that equation needs to be expanded to calculate
the fraction of $L$ that falls between pairs of vertical heights $z_a > z_b$. The actual
area amount of leaf area $A_l$ for an individual stem falling between those two heights
is simply the diffence in projected leaf area between the two heights:

$$
A_{l[a,b]} = \tilde{A}_{cp}(z_a) - \tilde{A}_{cp}(z_b)
$$

Given that information, the calculation of $f_{abs}$ becomes:

$$
f_{abs[a,b]} = 1 - e^{\left(-k\dfrac{L A_{l[a,b]}}{A_c}\right)}
$$

When $z_a = H$ and $z_b=0$, then $A_{l[a,b]} = A_c$ and hence simplifies to the original
equation.

The code below creates a simple example community containing a single cohort containing
a single stem and then calculates the light extinction profile down through the canopy.

```{code-cell} ipython3
# Create a simple community with a single stem
simple_pft = PlantFunctionalType(name="defaults", m=2, n=2)
simple_flora = Flora([simple_pft])
stem_dbh = np.array([0.5])
simple_stem = StemAllometry(stem_traits=simple_flora, at_dbh=stem_dbh)

# The total area is exactly the crown area
total_area = simple_stem.crown_area[0][0]

# Define a simple community
simple_community = Community(
    flora=simple_flora,
    cell_area=total_area,
    cell_id=1,
    cohorts=Cohorts(
        dbh_values=stem_dbh,
        n_individuals=np.array([1]),
        pft_names=np.array(["defaults"]),
    ),
)

# Get the canopy model for the simple case from the canopy top
# to the ground
hghts = np.linspace(simple_stem.stem_height[0][0], 0, num=101)[:, None]
simple_canopy = Canopy(
    community=simple_community,
    layer_heights=hghts,
)
```

As a simple check that the calculation across height layers is correct, the canopy
instance returns a vertical light extinction profile. The last value in this profile
should equal the whole canopy $f_{abs}$ calculated using the simple Beer-Lambert
equation and the PFT trait values.

```{code-cell} ipython3
print(simple_canopy.community_data.extinction_profile[-1])
```

```{code-cell} ipython3
print(1 - np.exp(-simple_pft.par_ext * simple_pft.lai))
```

The plot below shows:

1. The shape of crown radius for the stem (solid line) along with the projected leaf
   radius (dashed line). The leaf radius does not show the actual expected boundary of
   leaf area - which follows the crown - but is useful to visualise the displacement of
   leaf area on the same scale as the crown radius.
2. The vertical profile of the actual leaf area $A_{l[a,b]}$ between two height.
3. The resulting light absorption at each height.
4. The light extinction profile through the canopy.

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, sharey=True, figsize=(12, 6))

# Generate plot structures for stem profiles
ch, crown_radius = get_crown_xy(
    crown_profile=simple_canopy.crown_profile,
    stem_allometry=simple_community.stem_allometry,
    attr="crown_radius",
    two_sided=False,
)[0]

ch, projected_leaf_radius = get_crown_xy(
    crown_profile=simple_canopy.crown_profile,
    stem_allometry=simple_community.stem_allometry,
    attr="projected_leaf_radius",
    two_sided=False,
)[0]

ax1.plot(crown_radius, ch, color="red")
ax1.plot(projected_leaf_radius, ch, linestyle="--", color="red")
ax1.set_xlabel("Profile radius (m)")
ax1.set_ylabel("Vertical height (m)")

# Plot the leaf area between heights for stems
ax2.plot(simple_canopy.cohort_data.stem_leaf_area, hghts, color="red")
ax2.set_xlabel("Leaf area (m2)")

# Plot the fraction of light absorbed at different heights
ax3.plot(simple_canopy.cohort_data.f_abs, hghts, color="red")
ax3.set_xlabel("Light absorption fraction (-)")

# Plot the light extinction profile through the canopy.
ax4.plot(simple_canopy.community_data.extinction_profile, hghts, color="red")
ax4.set_xlabel("Cumulative light\nabsorption fraction (-)")
```

## Light extinction within a community

Within a community, the calculations above need to be modified to account for:

* the number of cohorts $n$,
* the number of individuals $i_{h}$ within each cohort,
* differences in the LAI $L_{h}$ and light extinction coefficients $k_{h}$ between
  cohorts,
* scaling LAI to the total area available to the community $A_T$ rather than the cohort
  specific crown area $A_h$.

Within the community, each cohort now requires a whole cohort  LAI component $L_H$,
which consists of the total leaf area index across individuals divided by the total
community area to give an average leaf area index across the available space:

$$
L_H = \frac{i_h L_h A_h}{A_T}
$$

The Beer-Lambert equation across the cohorts is then:

$$
f_{abs} = 1 - e^{\left(\sum\limits_{m=1}^{n}-k_{[m]} L_{H[m]}\right)}
        =  1 - \prod\limits_{m=1}^{n}e^{-k_{[m]}  L_{H[m]}}
$$

This equation can be adjusted as before to partition light absorption within vertical
layers and the implementation is demonstrated below using a simple community containing
two plant functional types:

* a shorter understory tree with a columnar canopy and no crown gaps
* a taller canopy tree with a top heavy canopy and more crown gaps

and then three cohorts:

* 7 saplings of the short PFT
* 3 larger stems of the short PFT
* 2 large stems of tall PFT

```{code-cell} ipython3
# Define PFTs
short_pft = PlantFunctionalType(
    name="short",
    h_max=15,
    m=1.5,
    n=1.5,
    par_ext=0.4,
    f_g=0,
    ca_ratio=380,
    lai=4,
)
tall_pft = PlantFunctionalType(
    name="tall", h_max=30, m=3, n=1.5, par_ext=0.6, f_g=0.2, ca_ratio=500
)

# Create the flora
flora = Flora([short_pft, tall_pft])

# Define a simply community with three cohorts
community = Community(
    flora=flora,
    cell_area=32,
    cell_id=1,
    cohorts=Cohorts(
        dbh_values=np.array([0.1, 0.20, 0.5]),
        n_individuals=np.array([7, 3, 2]),
        pft_names=np.array(["short", "short", "tall"]),
    ),
)

# Calculate the canopy profile across vertical heights
hghts = np.linspace(community.stem_allometry.stem_height.max(), 0, num=101)[:, None]
canopy = Canopy(community=community, layer_heights=hghts)
```

The plot below then shows a simplistic 2D representation of the community.

```{code-cell} ipython3
fig, ax = plt.subplots(ncols=1)

# Extract the crown profiles as XY arrays for plotting
profiles = get_crown_xy(
    crown_profile=canopy.crown_profile,
    stem_allometry=community.stem_allometry,
    attr="crown_radius",
    as_xy=True,
)

for idx, crown in enumerate(profiles):

    # Get spaced but slightly randomized stem locations
    n_stems = community.cohorts.n_individuals[idx]
    stem_locations = np.linspace(0, 10, num=n_stems) + np.random.normal(size=n_stems)

    # Plot the crown model for each stem
    for stem_loc in stem_locations:
        ax.add_patch(Polygon(crown + np.array([stem_loc, 0]), color="#00550055"))

ax.autoscale_view()
ax.set_aspect(1)
```

As before, we can verify that the cumulative light extinction at the bottom of the
vertical profile is equal to the expected value across the whole community.

```{code-cell} ipython3
# Calculate L_h for each cohort
cohort_lai = (
    community.cohorts.n_individuals
    * community.stem_traits.lai
    * community.stem_allometry.crown_area
) / community.cell_area

# Calculate 1 - e ^ -k L
print(1 - np.exp(np.sum(-community.stem_traits.par_ext * cohort_lai)))
```

```{code-cell} ipython3
print(canopy.community_data.extinction_profile[-1])
```

```{code-cell} ipython3
:tags: [hide-input]

cols = ["r", "b", "g"]

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=cols)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, sharey=True, figsize=(12, 6))

# Generate plot structures for stem profiles
crown_profile = get_crown_xy(
    crown_profile=canopy.crown_profile,
    stem_allometry=community.stem_allometry,
    attr="crown_radius",
    two_sided=False,
)

leaf_profile = get_crown_xy(
    crown_profile=canopy.crown_profile,
    stem_allometry=community.stem_allometry,
    attr="projected_leaf_radius",
    two_sided=False,
)

for (stem_rh, stem_cr), (stem_lh, stem_plr), col in zip(
    crown_profile, leaf_profile, cols
):
    ax1.plot(stem_cr, stem_rh, color=col)
    ax1.plot(stem_plr, stem_lh, linestyle="--", color=col)

ax1.set_xlabel("Profile radius (m)")
ax1.set_ylabel("Vertical height (m)")

# Plot the leaf area between heights for stems
ax2.plot(canopy.cohort_data.stem_leaf_area, hghts)
ax2.set_xlabel("Leaf area per stem (m2)")

# Plot the fraction of light absorbed at different heights
ax3.plot(canopy.cohort_data.f_abs, hghts, color="grey")
ax3.plot(1 - canopy.cohort_data.f_trans, hghts)
ax3.set_xlabel("Light absorption fraction (-)")

# Plot the light extinction profile through the canopy.
ax4.plot(canopy.community_data.extinction_profile, hghts, color="grey")
_ = ax4.set_xlabel("Cumulative light\nabsorption fraction (-)")
```

## Canopy closure and canopy gap fraction

:::{admonition} TODO

Need to work out how to include the gap fraction in the calculation of light extinction
because at the moment, the gap fraction in the PPA calculates the layer closure heights
accounting for that, but the LAI is not accounting for it so there is no shift in the
light extinction profile.

:::

In addition to calculating profiles from a provided sequence of vertical heights, the
canopy model also implements the calculation of canopy layers, following the perfect
plasticity approximation (PPA) {cite}`purves:2008a`. This model divides the vertical
structure of the canopy into discrete closed layers. The model assumes that plants are
always able to plastically arrange their crown within the broader canopy of the
community to maximise their crown size and fill the available space $A$. When the area
$A$ is filled, a new lower canopy layer is formed until all of the individual crown area
has been distributed across within the canopy.

A simple method for finding the first canopy closure height is to find a height $z^*_1$
at which the sum of crown projected area across all stems $N_s$ in a community equals
$A$:

$$
\sum_1^{N_s}{ A_p(z^*_1)} = A
$$

However, the canopy model also allows for modification by a community-level **canopy gap
fraction** ($f_G$) that captures the overall proportion of the canopy area that is left
unfilled by canopy. This gap fraction, capturing processes such as crown shyness,
describes the proportion of open sky visible from the forest floor. This gives the
following definition of the height of canopy layer closure ($z^*_l$) for a given canopy
layer $l = 1, ..., l_m$:

$$
\sum_1^{N_s}{ A_p(z^*_l)} = l A(1 - f_G)
$$

The set of heights $z^*$  can be found numerically by using a root solver to find values
of $z^*_l$ for $l = 1, ..., l_m$ that satisfy:

$$
\sum_1^{N_s}{ A_p(z^*_l)} - l A(1 - f_G) = 0
$$

The total number of layers $l_m$ in a canopy, where the final layer may not be fully
closed, can be found given the total crown area across stems as:

$$
l_m = \left\lceil \frac{\sum_1^{N_s}{A_c}}{ A(1 - f_G)}\right\rceil
$$

```{code-cell} ipython3
canopy_ppa = Canopy(community=community, canopy_gap_fraction=0 / 32, fit_ppa=True)
```

The `canopy_ppa.heights` attribute now contains the heights at which the PPA
layers close:

```{code-cell} ipython3
canopy_ppa.heights
```

And the final value in the canopy extinction profile still matches the expectation from
above:

```{code-cell} ipython3
print(canopy_ppa.community_data.extinction_profile[-1])
```

### Visualizing layer closure heights and areas

We can use the crown profile calculated for the previous canopy model to calculate a
whole community crown and leaf area profile for the community. For each height,
we calculate the sum of the product of stem projected area and the number of
individuals in each cohort.

```{code-cell} ipython3
# Calculate the total projected crown area across the community at each height
community_crown_area = np.nansum(
    canopy.crown_profile.projected_crown_area * community.cohorts.n_individuals,
    axis=1,
)

# Do the same for the projected leaf area
community_leaf_area = np.nansum(
    canopy.crown_profile.projected_leaf_area * community.cohorts.n_individuals,
    axis=1,
)
```

We can now plot community-wide $A_p(z)$ and $\tilde{A}_{cp}(z)$ profiles, and
superimpose the calculated $z^*_l$ values and the cumulative canopy area for each layer
to confirm that the calculated values coincide with the profile. Note here that the
total area at each closed layer height is omitting the community gap fraction.

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12, 6))

# Calculate the crown area at which each canopy layer closes.
closure_areas = np.arange(1, len(canopy_ppa.heights) + 1) * canopy.filled_community_area

# LH plot - projected leaf area with height.

# Add lines showing the canopy closure heights and closure areas.
for val in canopy_ppa.heights:
    ax1.axhline(val, color="red", linewidth=0.5, zorder=0)

for val in closure_areas:
    ax1.axvline(val, color="red", linewidth=0.5, zorder=0)

# Show the community projected crown area profile
ax1.plot(community_crown_area, canopy.heights, zorder=1, label="Crown area")
ax1.plot(
    community_leaf_area,
    canopy.heights,
    zorder=1,
    linestyle="--",
    color="black",
    linewidth=1,
    label="Leaf area",
)

# Add cumulative canopy area at top
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())
area_labels = [f"$A_{l + 1}$ = {z:.1f}" for l, z in enumerate(np.nditer(closure_areas))]
ax1_top.set_xticks(closure_areas)
ax1_top.set_xticklabels(area_labels, rotation=90)

ax1.set_ylabel("Vertical height ($z$, m)")
ax1.set_xlabel("Community-wide projected area (m2)")
ax1.legend(frameon=False)

# RH plot - light extinction
for val in canopy_ppa.heights:
    ax2.axhline(val, color="red", linewidth=0.5, zorder=0)

for val in canopy_ppa.community_data.extinction_profile:
    ax2.axvline(val, color="red", linewidth=0.5, zorder=0)

ax2.plot(canopy.community_data.extinction_profile, hghts)

ax2_top = ax2.twiny()
ax2_top.set_xlim(ax2.get_xlim())
extinction_labels = [
    f"$f_{{abs{l + 1}}}$ = {z:.3f}"
    for l, z in enumerate(np.nditer(canopy_ppa.community_data.extinction_profile))
]
ax2_top.set_xticks(canopy_ppa.community_data.extinction_profile)
ax2_top.set_xticklabels(extinction_labels, rotation=90)

ax2.set_xlabel("Light extinction (-)")

# Add z* values on the righthand axis
ax2_rhs = ax2.twinx()
ax2_rhs.set_ylim(ax2.get_ylim())
z_star_labels = [
    f"$z^*_{l + 1} = {val:.2f}$" for l, val in enumerate(np.nditer(canopy_ppa.heights))
]
ax2_rhs.set_yticks(canopy_ppa.heights.flatten())
_ = ax2_rhs.set_yticklabels(z_star_labels)
```

## Light allocation

<!-- markdownlint-disable  MD029 -->

In order to use the light extinction with the P Model, we need to calculate the fraction
of absorbed photosynthetically active radiation $f_{APAR}$ within each layer for each
cohort. These values can be multiplied by the canopy-top photosynthetic photon flux
density (PPFD) to give the actual light absorbed for photosynthesis.

The steps below show this partitioning process for the PPA layers calculated above.

1. Calculate the fraction of light transmitted $f_{tr}$ through each layer for each
   cohort. The two arrays below show the extinction coefficients for the PFT of each
   cohort and then the cohort LAI ($L_H$, columns) components within each layer (rows).
   The transmission through each component is then $f_{tr}=e^{-kL_H}$ and
   $f_{abs} = 1 - f_{tr}$ .

```{code-cell} ipython3
print("k = \n", community.stem_traits.par_ext, "\n")
print("L_H = \n", canopy_ppa.cohort_data.lai)
```

```{code-cell} ipython3
layer_cohort_f_tr = np.exp(-community.stem_traits.par_ext * canopy_ppa.cohort_data.lai)
print(layer_cohort_f_tr)
```

```{code-cell} ipython3
layer_cohort_f_abs = 1 - layer_cohort_f_tr
print(layer_cohort_f_abs)
```

   These matrices show that there is complete transmission ($f_{abs} = 0, f_{tr} = 1$)
   where a given stem has no leaf area within the layer but otherwise the leaves of each
   stem absorb some light.

2. Calculate the total transmission across cohorts within each layer, as the product of
   the individual cohort transmission within the layers, and then the absorption within
   each layer

```{code-cell} ipython3
layer_f_tr = np.prod(layer_cohort_f_tr, axis=1)
print(layer_f_tr)
```

```{code-cell} ipython3
layer_f_abs = 1 - layer_f_tr
print(layer_f_abs)
```

3. Calculate the transmission and extinction profiles through the layers as the
   cumulative product of light transmitted.

```{code-cell} ipython3
transmission_profile = np.cumprod(layer_f_tr)
print(transmission_profile)
```

```{code-cell} ipython3
extinction_profile = 1 - transmission_profile
print(extinction_profile)
```

4. Calculate the fraction of light transmitted through each each layer:

```{code-cell} ipython3
layer_fapar = -np.diff(transmission_profile, prepend=1)
print(layer_fapar)
```

5. Calculate the relative absorbance across cohort within each layer and then use this
   to partition the light absorbed in that layer across the cohorts:

```{code-cell} ipython3
cohort_fapar = (
    layer_cohort_f_abs / layer_cohort_f_abs.sum(axis=1)[:, None]
) * layer_fapar[:, None]
print(cohort_fapar)
```

6. Last, divide the cohort $f_{APAR}$ through by the number of individuals in each
   cohort to given the $f_{APAR}$ for each stem at each height.

```{code-cell} ipython3
stem_fapar = cohort_fapar / community.cohorts.n_individuals
print(stem_fapar)
```
