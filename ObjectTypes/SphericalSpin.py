"""
Spherical spins are real valued variables with a global constraint on overall norm
"""

from ..Objects import Object
from numpy import array
from numpy.linalg import norm


def makeObject(Value, NormConstraint, Group = None):
    """
    Make a spherical spin object
    :param Value: Spin real value
    :param NormConstraint: The spherical constraint on all spins of this group
    :param Group: Group number that enables different groups of spins with different constraints
    :return: Object
    """

    Parameters = dict()
    if Group == None :
        Type = 'Spherical Spin'
        ParKeys = []
    else:
        assert int(Group) > 0
        Type = 'Spherical Spin ' + str(int(Group))
        Parameters['Group'] = int(Group)
        ParKeys = ['Group']

    Variables = dict()
    assert type(Value) == float
    Variables['Value'] = Value
    VarKeys = ['Value']

    assert (type(NormConstraint) == float) and (NormConstraint > 0)
    GlobalConstraint = dict()
    GlobalConstraint['Kinds'] = ['NormConstraint']
    GlobalConstraint['Keys'] = [['Value']]
    GlobalConstraint['Parameters'] = [[NormConstraint]]

    Obj = Object(Type, Variables, Parameters, VarKeys, ParKeys, GlobalConstraint)
    return Obj


def Random_SphericalSpins(N,NormConstraint,Group=None):
    from numpy.random import randn

    Objects = []
    Val = randn(N)
    Val = Val/norm(Val) * NormConstraint
    for i in range(N):
        Objects.append(makeObject(float(Val[i]), float(NormConstraint), Group))

    return Objects