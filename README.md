![](docs/source/_static/STREAM6.png)

# **S**ystem **T**hermohydraulics for **R**eactor **E**valuation, **A**nalysis & **M**odeling

In a sentence: ***STREAM is a versatile, easily extensible, modern nuclear reactor system code***. 

A `System Code` is a tool enabling the simulation of nuclear reactor heat evacuation during normal operation and postulated accidents, 
which is done approximately without employing very expensive higher fidelity tools as CFD (computational fluid dynamics) codes and neutron transport.

In a few more words, STREAM is:
#### 1. A general purpose graph-based solver for large systems of coupled differential and algebraic equations:
```math
M\frac{d\vec{y}}{dt} = \vec{F}(\vec{y}, t)
```
This equation is described by designating subsections as separate (coupled but independent) calculations. 
Those calucaltions are viewed as nodes in a directed graph whose edges are the coupled variables.

#### 2. A library of thermal-hydraulic components and correlations
STREAM supports the following capabilities:

* Single phase, one dimensional coolant channels.
* Point Neutronics and decay heat computation.
* Reactor Protection System events and triggers.
* Support for a general incompressible flow graph through Kirchhoff's Laws:
```math
\sum_{\text{loop}}\Delta p = 0; \sum_{\text{junction}}\dot{m}=0
```
* Cartesian and cylindrical heat structures with 2D heat diffusion.
* Subcooled Boiling heat transfer support.
* Flow regime support (**turbulent, laminar, and free convection**)
* Many other thermohydraulic components: Ideal pump, heat exchanger, friction and local regime dependent pressure drops, flapper
* Uncertainty Quantification through Uncertainty Propagation (which can be computed in a distributed fashion through DASK)
