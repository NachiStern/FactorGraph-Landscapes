# FactorGraph-Landscapes

##General physically inspired energy landscape builder + Optimization and (Machine-)Learning

FactorGraphh-Landscapes is a python program made for seamless construction and expolration of general physically inspired energy landscapes. This is achieved by partitioning the physical variables (particle positions, spins, etc.) and pyhsical parameters (bond strengths, couplings etc.) to two groups, *objects* and *interactions*, respectively. The user is free to include any number of objecct and interaction types, and connect them in any way that respects the interaction rules of the specific problem. Thus it is possible to create extremely general potential surfaces, with arbitrarily complex *n*-body type interactions. The energy is computed as a sum over the interaction terms.

Written in Cython-boosted Python, the code is meant to facilitate exploration of general energy landscapes. It ths support independent (and mutual) optimization of the object and interaction terms. Optimization of the object terms (physical variables) is supposed akin to minimizing the statistical free energy and finding stable points of the system, while interaction optimization is closely related to statistical (machine-)learning.





