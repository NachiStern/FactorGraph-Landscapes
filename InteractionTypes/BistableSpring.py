## Weird "spring"-like interactions between 2 mechanical nodes with 2 minima

from numpy import eye, sqrt, dot, shape, reshape, array, empty_like
from numdifftools import Hessian
from ..Interactions import Interaction

def makeInteraction(Dim,X0,L0,K1,K2,Metric):
    Type = 'BistableSpring'
    Variables = dict()
    Parameters = dict()

    assert int(Dim) > 0
    Parameters['SpatialDimension'] = int(Dim)

    assert X0 >= 0
    Variables['X0'] = X0

    assert (L0 >= 0) and (L0 <= X0)
    Variables['L0'] = L0

    assert K1 >= 0
    Variables['K1'] = K1

    assert K2 >= 0
    Variables['K2'] = K2

    VarKeys = ['X0', 'L0', 'K1', 'K2']

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
                      BareHessObj = BareHessObj, BareHessInt = BareHessInt)
    return Int


def EnergyFunc(ObjVar, IntVar, ObjPar, IntPar):
    X0 = IntVar['X0']
    L0 = IntVar['L0']
    K1 = IntVar['K1']
    K2 = IntVar['K2']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))

    E0 = 0.5 * K2 * (L0**2 - 0.5 * K2/K1)
    Energy = 0.25 * K1 * (Dist - X0)**4  -  0.5 * K2 * (Dist - X0 - L0)*(Dist - X0 + L0)  -  E0
    return Energy


def GradObjFunc(ObjVar, IntVar, ObjPar, IntPar, wrtObject):
    X0 = IntVar['X0']
    #L0 = IntVar['L0']
    K1 = IntVar['K1']
    K2 = IntVar['K2']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    assert (wrtObject < len(ObjVar)) and (wrtObject >= 0)

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))

    Gradient = dict()
    Gradient['Position'] = (K1*(Dist-X0)**3 - K2*(Dist-X0)) * (dot(Metric, Diff)/Dist) * (-1) ** (wrtObject - 1)
    return Gradient


def GradIntFunc(ObjVar, IntVar, ObjPar, IntPar):
    X0 = IntVar['X0']
    L0 = IntVar['L0']
    K1 = IntVar['K1']
    K2 = IntVar['K2']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))

    Gradient = dict()
    Gradient['X0'] = - K1*(Dist-X0)**3 + K2*(Dist-X0)
    Gradient['L0'] = K2*L0
    Gradient['K1'] = 0.25 * (Dist - X0)**4
    Gradient['K2'] = 0.5 *  (Dist - X0 - L0)*(Dist - X0 + L0)
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

    X0 = IntVar[0]
    L0 = IntVar[1]
    K1 = IntVar[2]
    K2 = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    dist = sqrt(dot(dot(IntPar[1], diff), diff))

    energy = 0.25 * K1 * (dist - X0)**4  -  0.5 * K2 * (dist - X0 - L0)*(dist - X0 + L0)  -  \
             0.5 * K2 * (L0**2 - 0.5 * K2/K1)
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

    X0 = IntVar[0]
    L0 = IntVar[1]
    K1 = IntVar[2]
    K2 = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))

    energy = 0.25 * K1 * (dist - X0)**4  -  0.5 * K2 * (dist - X0 - L0)*(dist - X0 + L0)  -  \
             0.5 * K2 * (L0**2 - 0.5 * K2/K1)

    gradobj = empty_like(ObjVar)
    gradobj[:dimI] = - (K1*(dist-X0)**3 - K2*(dist-X0)) * (metricDot/dist)
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

    X0 = IntVar[0]
    L0 = IntVar[1]
    K1 = IntVar[2]
    K2 = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))

    energy = 0.25 * K1 * (dist - X0) ** 4 - 0.5 * K2 * (dist - X0 - L0) * (dist - X0 + L0) - \
             0.5 * K2 * (L0 ** 2 - 0.5 * K2 / K1)

    gradint = empty_like(IntVar)
    gradint[0] = - K1*(dist-X0)**3 + K2*(dist-X0)
    gradint[1] = K2*L0
    gradint[2] = 0.25 * (dist - X0)**4
    gradint[3] = 0.5 *  (dist - X0 - L0)*(dist - X0 + L0)
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

    HessFunc = Hessian(BareEnergy)
    HessObj = HessFunc(ObjVar, IntVar, ObjPar, IntPar)
    return HessObj


def BareEnergyInt(IntVar, ObjVar, ObjPar, IntPar):
    """
    Bare form of energy function. Variables and parameters are ordered as:
    IntVar = (RestLength, Stiffness)
    ObjVar = ([PosObj1], [PosObj2]) as one list
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    """
    dimI = int(IntPar[0])

    X0 = IntVar[0]
    L0 = IntVar[1]
    K1 = IntVar[2]
    K2 = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    dist = sqrt(dot(dot(IntPar[1], diff), diff))
    energy = 0.25 * K1 * (dist - X0)**4  -  0.5 * K2 * (dist - X0 - L0)*(dist - X0 + L0)  -  \
             0.5 * K2 * (L0**2 - 0.5 * K2/K1)
    return energy


def BareHessInt(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object Hessian function. Variables and parameters are ordered as:
    ObjVar = ([PosObj1], [PosObj2]) as one list
    IntVar = (RestLength, Stiffness)
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    :return: Hessian matrix w.r.t interaction variables
    """

    HessFunc = Hessian(BareEnergyInt)
    HessInt = HessFunc(IntVar, ObjVar, ObjPar, IntPar)
    return HessInt

def Random_Bistable_Springs(N,Dim):
    #from numpy.random import randn

    Interactions = []
    K1 = 1
    K2 = 1
    X0 = 2
    L0 = 1
    Metric = 'Euclidean'
    for i in range(N):
        Interactions.append(makeInteraction(Dim, X0, L0, K1, K2, Metric))

    return Interactions