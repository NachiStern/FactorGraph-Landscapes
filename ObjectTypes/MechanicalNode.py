"""
Mechanical nodes are spatial points in a d-dim vector space.
"""

from ..Objects import Object
from numpy import array

def makeObject(Dim,Position,Bounds=None):
    """
    Make a mechanical node object
    :param Dim: Spatial dimension
    :param Position: Position in space
    :param Bounds: 2 values for min/max for each dimension (i.e. 2d looks like [[minX,maxX],[minY,maxY]])
    :return: Object with these features
    """
    Type = 'Mechanical Node'
    Parameters = dict()
    Dim = int(Dim)
    Parameters['SpatialDimension'] = Dim
    ParKeys = ['SpatialDimension']
    Variables = dict()
    Variables['Position'] = array(Position)
    VarKeys = ['Position']

    if Bounds != None:
        DBounds = {'Position': []}
        assert len(Bounds) == Dim
        for i in range(len(Bounds)):
            assert len(Bounds[i]) == 2
            DBounds['Position'].append(Bounds[i])
        Bounds = {'Position': DBounds['Position']}

    Obj = Object(Type, Variables, Parameters, VarKeys, ParKeys, Bounds = Bounds)
    return Obj


def Random_MechanicalNodes(N,Dim):
    """
    Make N nodes of dimension Dim with a gaussian distribution N(0,1)
    :param N:
    :param Dim:
    :return:
    """
    from numpy.random import randn

    assert (type(N)==int) and (N>=1)
    assert (type(Dim)==int) and (Dim>=1)

    Objects = []
    Pos = randn(N,Dim)
    for i in range(N):
        Objects.append(makeObject(Dim, Pos[i]))

    return Objects


def Uniform_MechanicalNodes(N,Dim,BoxBounds=[], AvoidSymmetries=False):
    """
    Make N nodes of dimension Dim uniformly distributed in the space [0., Bounds[i]]
    :param N:
    :param Dim:
    :param BoxBounds: Sequence of upper bounds in each dimension
    :return:
    """
    from numpy.random import rand
    from numpy import ones, sign

    assert (type(N) == int) and (N >= 1)
    assert (type(Dim) == int) and (Dim >= 1)

    eps = 1.e-3

    if not len(BoxBounds):   # if no box bounds given set the uniform distributions U(0,1)
        BoxBounds = ones(Dim)

    if AvoidSymmetries: # find origin
        X0 = array(BoxBounds) / 2.

    Objects = []
    Pos = array([ rand(N) * BoxBounds[i] for i in range(Dim) ])
    Pos = Pos.T
    for i in range(N):
        if AvoidSymmetries:
            if i < Dim: # first d-particles are boundsd to some subspaces to avoid trivial symmetries
                Posi = Pos[i]
                bounds = []
                for d in range(Dim-i):
                    bounds.append([X0[d]-eps, X0[d]+eps])
                    Posi[d] = X0[d]
                for d in range(Dim-i, min(Dim-i+1, Dim)):
                    bounds.append([X0[d], None])
                    Posi[d] = X0[d] + (Posi[d]-X0[d]) * sign(Posi[d]-X0[d])
                for d in range(Dim-i+1,Dim):
                    bounds.append([None, None])
                Objects.append(makeObject(Dim, Posi, bounds))
            elif i == Dim: # rule out one final reflection symmetry
                Posi = Pos[i]
                bounds = []
                Posi[0] = X0[0] + (Posi[0]-X0[0]) * sign(Posi[0]-X0[0])
                bounds.append([X0[0], None])
                for d in range(1,Dim):
                    bounds.append([None, None])
                Objects.append(makeObject(Dim, Posi, bounds))
            else:
                bounds = []
                for d in range(Dim):
                    bounds.append([None,None])
                Objects.append(makeObject(Dim, Pos[i]))
        else:
            Objects.append(makeObject(Dim, Pos[i]))

    return Objects