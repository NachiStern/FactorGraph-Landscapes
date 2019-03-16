from numpy import eye, sqrt, dot, shape, array, empty_like, empty
from ..Interactions import Interaction

def makeInteraction(Dim,RestLength,Stiffness,Metric, RestLengthBounds = [0.,10.], StiffnessBounds = [1.,10.]):
    Type = 'Spring'
    Variables = dict()
    Parameters = dict()

    assert int(Dim) > 0
    Parameters['SpatialDimension'] = int(Dim)

    assert RestLength >= 0
    Variables['RestLength'] = RestLength

    assert Stiffness >= 0
    Variables['Stiffness'] = Stiffness

    VarKeys = ['RestLength', 'Stiffness']

    Bounds = dict()
    Bounds['RestLength'] = RestLengthBounds
    Bounds['Stiffness'] = StiffnessBounds

    if Metric == 'Euclidean':
        Metric = eye(Dim)
    if shape(Metric) != (Dim, Dim):
        Metric = eye(Dim)  # if metric is not in right shape force Euclidean
    Parameters['Metric'] = Metric

    ParKeys = ['SpatialDimension', 'Metric']

    InputObjects = [[['Mechanical Node'], 2]]

    Int = Interaction(Type, Variables, Parameters, VarKeys, ParKeys, InputObjects,
                      EnergyFunc, BareEnergy,
                      GradObj = GradObjFunc, GradInt = GradIntFunc,
                      BareGradObj = BareGradObj, BareGradInt = BareGradInt,
                      BareHessObj = BareHessObj, BareHessInt = BareHessInt,
                      Bounds = Bounds)
    return Int


def EnergyFunc(ObjVar, IntVar, ObjPar, IntPar):
    RestLength = IntVar['RestLength']
    Stiffness = IntVar['Stiffness']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))
    Energy = 0.5 * Stiffness * (Dist - RestLength) ** 2
    return Energy


def GradObjFunc(ObjVar, IntVar, ObjPar, IntPar, wrtObject):
    RestLength = IntVar['RestLength']
    Stiffness = IntVar['Stiffness']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    assert (wrtObject < len(ObjVar)) and (wrtObject >= 0)

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))

    Gradient = dict()
    Gradient['Position'] = Stiffness * (1. - RestLength / Dist) * dot(Metric, Diff) * (-1) ** (wrtObject - 1)
    return Gradient


def GradIntFunc(ObjVar, IntVar, ObjPar, IntPar):
    RestLength = IntVar['RestLength']
    Stiffness = IntVar['Stiffness']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))

    Gradient = dict()
    Gradient['RestLength'] = Stiffness * (RestLength - Dist)
    Gradient['Stiffness'] = 0.5 * (Dist - RestLength) ** 2
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

    restLength = IntVar[0]
    stiffness = IntVar[1]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    dist = sqrt(dot(dot(IntPar[1], diff), diff))
    energy = 0.5 * stiffness * (dist - restLength) ** 2
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

    restLength = IntVar[0]
    stiffness = IntVar[1]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))

    energy = 0.5 * stiffness * (dist - restLength) ** 2

    gradobj = empty_like(ObjVar)
    gradobj[:dimI] = - stiffness * (1. - restLength / dist) * metricDot
    gradobj[dimI:] = - gradobj[:dimI]
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

    restLength = IntVar[0]
    stiffness = IntVar[1]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    dist = sqrt(dot(dot(IntPar[1], diff), diff))

    energy = 0.5 * stiffness * (dist - restLength) ** 2

    gradint = empty_like(IntVar)
    gradint[0] = stiffness * (restLength - dist)
    gradint[1] = 0.5 * (dist - restLength) ** 2
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

    restLength = IntVar[0]
    stiffness = IntVar[1]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    metric = IntPar[1]
    metricDot = dot(metric, diff)
    dist = sqrt(dot(metricDot, diff))

    H1 = stiffness * (1. - restLength / dist) * metric + stiffness * (restLength / dist**3) * dot(metricDot, metricDot)
    HessObj = empty([len(ObjVar), len(ObjVar)])
    HessObj[dimI:,dimI:] = H1
    HessObj[:dimI,:dimI] = H1
    HessObj[dimI:,:dimI] = -H1
    HessObj[:dimI,dimI:] = -H1
    return HessObj


def BareHessInt(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object Hessian function. Variables and parameters are ordered as:
    ObjVar = ([PosObj1], [PosObj2]) as one list
    IntVar = (RestLength, Stiffness)
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    :return: Hessian matrix w.r.t interaction variables
    """
    dimI = int(IntPar[0])

    restLength = IntVar[0]
    stiffness = IntVar[1]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    metric = IntPar[1]
    metricDot = dot(metric, diff)
    dist = sqrt(dot(metricDot, diff))

    H1 = stiffness * (1. - restLength / dist) * metric + stiffness * (restLength / dist**3) * dot(metricDot, metricDot)
    HessInt = empty([len(IntVar), len(IntVar)])
    HessInt[0,0] = stiffness
    HessInt[1,0] = RestLength - dist
    HessInt[0,1] = RestLength - dist
    HessInt[1,1] = 0.
    return HessInt


def Random_Springs(N,Dim):
    #from numpy.random import randn

    Interactions = []
    Stiffness = 1
    RestLength = 0.3
    Metric = 'Euclidean'
    for i in range(N):
        Interactions.append(makeInteraction(Dim, RestLength, Stiffness, Metric))

    return Interactions