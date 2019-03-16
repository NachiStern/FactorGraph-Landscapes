## Pseudo Lennard Jones potential

from numpy import eye, sqrt, dot, shape, reshape, array, empty_like, zeros, log
from numdifftools import Hessian
from ..Interactions import Interaction

def makeInteraction(Dim,r0,a,b,k,Metric='Euclidean'):
    Type = 'Pseudo Lennard-Jones'
    Variables = dict()
    Parameters = dict()

    assert int(Dim) > 0
    Parameters['SpatialDimension'] = int(Dim)

    assert r0 >= 0
    Variables['r0'] = r0

    assert (b > 0) and (a > b)
    Variables['a'] = a
    Variables['b'] = b

    assert k >= 0
    Variables['k'] = k

    VarKeys = ['r0', 'a', 'b', 'k']

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
    r0 = IntVar['r0']
    a = IntVar['a']
    b = IntVar['b']
    k = IntVar['k']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))

    E0 = k * (float(b)/a - 1.)
    Energy = k * (float(b)/a * (r0/Dist)**a - (r0/Dist)**b) - E0
    return Energy


def GradObjFunc(ObjVar, IntVar, ObjPar, IntPar, wrtObject):
    r0 = IntVar['r0']
    a = IntVar['a']
    b = IntVar['b']
    k = IntVar['k']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    assert (wrtObject < len(ObjVar)) and (wrtObject >= 0)

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))

    Gradient = dict()
    Gradient['Position'] = k * float(b) * r0 / Dist**3 * ((r0/Dist)**(a-1) - (r0/Dist)**(b-1)) * dot(Metric, Diff) * (-1) ** wrtObject
    return Gradient


def GradIntFunc(ObjVar, IntVar, ObjPar, IntPar):
    r0 = IntVar['r0']
    a = IntVar['a']
    b = IntVar['b']
    k = IntVar['k']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))

    Gradient = dict()
    Gradient['r0'] = k*float(b)/Dist * ((r0/Dist)**(a-1.) - (r0/Dist)**(b-1.))
    Gradient['a'] = k*float(b)/a * (r0/Dist)**a * (log(a) - 1./a)
    Gradient['b'] = k * (1./a * (r0/Dist)**a - log(b) * (r0/Dist)**b)
    Gradient['k'] = float(b)/a * (r0/Dist)**a - (r0/Dist)**b
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
    #dimO1 = ObjPar[0]
    #dimO2 = ObjPar[1]
    #assert (dimI == dimO1) and (dimI == dimO2) and (dimO1 == dimO2)
    #metric = reshape(IntPar[1:], [dimI, dimI])
    #p1 = array(ObjVar[:dimI])
    #p2 = array(ObjVar[dimI:])

    r0 = IntVar[0]
    a = IntVar[1]
    b = IntVar[2]
    k = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    #dist = sqrt(dot(dot(metric, diff), diff))
    dist = sqrt(dot(dot(IntPar[1], diff), diff))
    energy = k * (float(b)/a * (r0/dist)**a - (r0/dist)**b) - k * (float(b)/a - 1.)
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
    #dimO1 = ObjPar[0]
    #dimO2 = ObjPar[1]
    #assert (dimI == dimO1) and (dimI == dimO2) and (dimO1 == dimO2)
    #metric = reshape(IntPar[1:], [dimI, dimI])
    #p1 = array(ObjVar[:dimI])
    #p2 = array(ObjVar[dimI:])

    r0 = IntVar[0]
    a = IntVar[1]
    b = IntVar[2]
    k = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    # dist = sqrt(dot(dot(metric, diff), diff))
    # metricDot = dot(IntPar[1:].reshape([dimI, dimI]), diff)
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))

    energy = k * (float(b) / a * (r0 / dist) ** a - (r0 / dist) ** b) - k * (float(b) / a - 1.)

    gradobj = empty_like(ObjVar)
    gradobj[:dimI] = - k * float(b) * r0 * ((r0/dist)**(a-1) - (r0/dist)**(b-1)) * (metricDot/dist**3)
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

    r0 = IntVar[0]
    a = IntVar[1]
    b = IntVar[2]
    k = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    # metricDot = dot(IntPar[1:].reshape([dimI, dimI]), diff)
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))

    energy = k * (float(b) / a * (r0 / dist) ** a - (r0 / dist) ** b) - k * (float(b) / a - 1.)

    gradint = empty_like(IntVar)
    gradint[0] = k / dist * (float(b) / a * (r0 / dist) ** a - (r0 / dist) ** b)
    gradint[1] = k * float(b) / a * (r0 / dist) ** a * (log(a) - 1. / a)
    gradint[2] = k * (1. / a * (r0 / dist) ** a - log(b) * (r0 / dist) ** b)
    gradint[3] = float(b) / a * (r0 / dist) ** a - (r0 / dist) ** b
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

    r0 = IntVar[0]
    a = IntVar[1]
    b = IntVar[2]
    k = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    dist = sqrt(dot(dot(IntPar[1:].reshape([dimI,dimI]), diff), diff))
    energy = k * (float(b) / a * (r0 / dist) ** a - (r0 / dist) ** b) - k * (float(b) / a - 1.)
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


def ConnectStablyByDistance(Objs, maxDist, params = [], Metric = []):
    """
    Connect all mechanical nodes close by with PLJ "springs"
    :param Objects: List of mechanical nodes
    :param maxDist: Maximum distance to connected nodes
    :param params: A list of interaction parameters for all interactions [a,b,k]
    :return I, A: Interaction List, Adjacency Matrix
    """

    Dim = Objs[0].Parameters['SpatialDimension']
    lo = len(Objs)

    for i in range(lo):
        assert Objs[i].Type == 'Mechanical Node'
        assert Objs[i].Parameters['SpatialDimension'] == Dim

    if not len(params):     # default is standard LJ potential
        params = [12., 6., 1.]

    if Metric == 'Euclidean':
        Metric = eye(Dim)
    if shape(Metric) != (Dim, Dim):
        Metric = eye(Dim)  # if metric is not in right shape force Euclidean

    Interactions = []
    Adjacency = []

    for i in range(lo):
        P1 = Objs[i].Variables['Position']
        for j in range(i+1,lo):
            P2 = Objs[j].Variables['Position']
            Diff = P2 - P1
            Dist = sqrt(dot(dot(Metric, Diff), Diff))

            if Dist <= maxDist :  # Connect the two objects
                Interactions.append( makeInteraction(Dim, Dist, params[0], params[1], params[2], Metric) )
                A = zeros(lo)
                A[i] = 1
                A[j] = 1
                Adjacency.append(A)

    Adjacency = array(Adjacency)
    return Interactions, Adjacency