"""
Scalar values are, well, scalar values
"""

from ..Objects import Object
from numpy import array

def makeObject(Value, Group = None):
    """
    Make a spherical spin object
    :param Value: real value
    :param Group: Group number that enables different groups objects
    :return: Object
    """

    Parameters = dict()
    if Group == None :
        Type = 'Scalar Value'
        ParKeys = []
    else:
        assert int(Group) > 0
        Type = 'Scalar Value ' + str(int(Group))
        Parameters['Group'] = int(Group)
        ParKeys = ['Group']

    Variables = dict()
    assert type(Value) == float
    Variables['Value'] = Value
    VarKeys = ['Value']

    Obj = Object(Type, Variables, Parameters, VarKeys, ParKeys)
    return Obj


def Random_Normal_Value(N, NormConstraint, Group = None):
    from numpy.random import randn

    Objects = []
    Val = randn(N)
    Val = Val/norm(Val) * NormConstraint
    for i in range(N):
        Objects.append(makeObject(float(Val[i]), Group))

    return Objects