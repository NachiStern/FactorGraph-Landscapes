## Unary interaction attaching a mechanical node to a point in space with a 0-length spring

from numpy import eye, sqrt, dot, shape, reshape, array, empty_like, empty, ndarray, size
from ..Interactions import Interaction

def makeInteraction(Dim,X0,Stiffness,Metric):
    Type = 'Spatial Pinning'
    Variables = dict()
    Parameters = dict()

    assert int(Dim) > 0
    Dim = int(Dim)
    Parameters['SpatialDimension'] = Dim

    assert (type(X0) == ndarray) and (size(X0) == Parameters['SpatialDimension'])
    Variables['X0'] = X0

    assert Stiffness >= 0
    Variables['Stiffness'] = Stiffness

    VarKeys = ['X0', 'Stiffness']

    if Metric == 'Euclidean':
        Metric = eye(Dim)
    if shape(Metric) != (Dim, Dim):
        Metric = eye(Dim)  # if metric is not in right shape force Euclidean
    Parameters['Metric'] = Metric

    ParKeys = ['SpatialDimension', 'Metric']

    InputObjects = [[['Mechanical Node'], 1]]

    Int = Interaction(Type, Variables, Parameters, VarKeys, ParKeys, InputObjects,
                      EnergyFunc, BareEnergy,
                      GradObj = GradObjFunc, GradInt = GradIntFunc,
                      BareGradObj = BareGradObj, BareGradInt = BareGradInt,
                      BareHessObj = BareHessObj, BareHessInt = BareHessInt)
    return Int


def EnergyFunc(ObjVar, IntVar, ObjPar, IntPar):
    X0 = IntVar['X0']
    Stiffness = IntVar['Stiffness']
    Metric = IntPar['Metric']

    PosObj = ObjVar[0]['Position']

    Diff = PosObj - X0
    Dist = sqrt(dot(dot(Metric, Diff), Diff))
    Energy = 0.5 * Stiffness * Dist ** 2
    return Energy


def GradObjFunc(ObjVar, IntVar, ObjPar, IntPar, wrtObject):
    X0 = IntVar['X0']
    Stiffness = IntVar['Stiffness']
    Metric = IntPar['Metric']

    PosObj = ObjVar[0]['Position']

    assert (wrtObject < len(ObjVar)) and (wrtObject >= 0)

    Diff = PosObj - X0
    #Dist = sqrt(dot(dot(Metric, Diff), Diff))

    Gradient = dict()
    Gradient['Position'] = Stiffness  * dot(Metric, Diff)
    return Gradient


def GradIntFunc(ObjVar, IntVar, ObjPar, IntPar):
    X0 = IntVar['X0']
    Stiffness = IntVar['Stiffness']
    Metric = IntPar['Metric']

    PosObj = ObjVar[0]['Position']

    Diff = PosObj - X0
    Dist = sqrt(dot(dot(Metric, Diff), Diff))

    Gradient = dict()
    Gradient['X0'] = - Stiffness  * dot(Metric, Diff)
    Gradient['Stiffness'] = 0.5 * Dist ** 2
    return Gradient


def BareEnergy(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of energy function. Variables and parameters are ordered as:
    ObjVar = ([PosObj1], [PosObj2]) as one list
    IntVar = (RestLength, Stiffness)
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    """
    dimI = int(IntPar[0])

    stiffness = IntVar[dimI]

    diff = ObjVar[:dimI] - IntVar[:dimI]  # x - X0
    dist = sqrt(dot(dot(IntPar[1], diff), diff))
    energy = 0.5 * stiffness * dist ** 2
    return energy


def BareGradObj(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object gradient function. Variables and parameters are ordered as:
    ObjVar = ([PosObj1], [PosObj2]) as one list
    IntVar = (RestLength, Stiffness)
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    :return: Energy and gradient w.r.t object variables
    """
    dimI = int(IntPar[0])

    stiffness = IntVar[dimI]

    diff = ObjVar[:dimI] - IntVar[:dimI]  # x - X0
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))

    energy = 0.5 * stiffness * dist ** 2

    gradobj = empty_like(ObjVar)
    gradobj[:dimI] = stiffness * metricDot
    return energy, gradobj


def BareGradInt(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object gradient function. Variables and parameters are ordered as:
    ObjVar = ([PosObj1], [PosObj2]) as one list
    IntVar = (RestLength, Stiffness)
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    :return: Energy and gradient w.r.t interaction variables
    """
    dimI = int(IntPar[0])

    stiffness = IntVar[dimI]

    diff = ObjVar[:dimI] - IntVar[:dimI]  # x - X0
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))

    energy = 0.5 * stiffness * dist ** 2

    gradint = empty_like(IntVar)
    gradint[0:dimI] = - stiffness * metricDot
    gradint[1] = 0.5 * dist ** 2
    return energy, gradint


def BareHessObj(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object Hessian function. Variables and parameters are ordered as:
    ObjVar = ([PosObj1], [PosObj2]) as one list
    IntVar = (RestLength, Stiffness)
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    :return: Hessian matrix w.r.t object variables
    """
    dimI = int(IntPar[0])

    stiffness = IntVar[dimI]

    HessObj = stiffness * IntPar[1]
    return HessObj


def BareHessInt(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object Hessian function. Variables and parameters are ordered as:
    ObjVar = ([PosObj1], [PosObj2]) as one list
    IntVar = (RestLength, Stiffness)
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    :return: Hessian matrix w.r.t object variables
    """
    dimI = int(IntPar[0])

    stiffness = IntVar[dimI]

    diff = ObjVar[:dimI] - IntVar[:dimI]  # x - X0
    metric = IntPar[1]
    metricDot = dot(metric, diff)

    HessInt = empty([len(IntVar), len(IntVar)])
    HessInt[:dimI,:dimI] = stiffness * metric
    HessInt[:dimI,dimI]  = - metricDot
    HessInt[dimI,:dimI]  = - metricDot
    HessInt[dimI,dimI]   = 0.
    return HessInt
