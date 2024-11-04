# FactorGraph-Landscapes

## General physically inspired energy landscape builder + Optimization and (Machine-)Learning

FactorGraphh-Landscapes is a python program made for seamless construction and expolration of general physically inspired energy landscapes. This is achieved by partitioning the physical variables (particle positions, spins, etc.) and pyhsical parameters (bond strengths, couplings etc.) to two groups, *objects* and *interactions*, respectively. The user is free to include any number of objecct and interaction types, and connect them in any way that respects the interaction rules of the specific system. Thus it is possible to create extremely general potential surfaces, with arbitrarily complex *n*-body type interactions. The energy is computed as a sum over the interaction terms.

Written in Cython-boosted Python, the code is meant to facilitate exploration of general energy landscapes. It supports independent (and mutual) optimization of the object and interaction terms. Optimization of the object terms (physical variables) is akin to minimizing the free energy and finding stable points of the system, while interaction optimization is closely related to computational (machine-)learning.


## MIT License

### Copyright (c) 2019 Menachem Stern

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


