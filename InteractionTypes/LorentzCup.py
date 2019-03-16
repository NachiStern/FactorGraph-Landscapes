## Lorentzian cup with short range repulsion

from numpy import eye, sqrt, dot, shape, reshape, array, empty_like, zeros, log, ctypeslib
from numpy cimport ndarray, int
from libc.math cimport sqrt as csqrt
cimport cython

from numdifftools import Hessian
from ..Interactions import Interaction

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(True)


def makeInteraction(Dim,kA,r0,Sigma,kR,rR,aR,Metric='Euclidean'):
    Type = 'Lorentzian Cup'
    Variables = dict()
    Parameters = dict()

    assert int(Dim) > 0
    Dim = int(Dim)
    Parameters['SpatialDimension'] = Dim

    assert (rR >= 0) and (r0 > rR)
    Variables['r0'] = r0
    Variables['rR'] = rR

    assert (Sigma > 0)
    Variables['Sigma'] = Sigma

    assert (aR > 0)
    Variables['aR'] = aR

    assert kR >= 0
    Variables['kR'] = kR

    assert kA > 0
    Variables['kA'] = kA

    VarKeys = ['kA', 'r0', 'Sigma', 'kR', 'rR', 'aR']

    if type(Metric) == str:
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
    rR = IntVar['rR']
    S = IntVar['Sigma']
    aR = IntVar['aR']
    kA = IntVar['kA']
    kR = IntVar['kR']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))
    D = Dist - r0

    Energy = kR * (rR/Dist)**aR + kA * D**2. / (D**2. + S**2.)
    return Energy


def GradObjFunc(ObjVar, IntVar, ObjPar, IntPar, wrtObject):
    r0 = IntVar['r0']
    rR = IntVar['rR']
    S = IntVar['Sigma']
    aR = IntVar['aR']
    kA = IntVar['kA']
    kR = IntVar['kR']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    assert (wrtObject < len(ObjVar)) and (wrtObject >= 0)

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))
    D = Dist - r0

    Gradient = dict()
    Gradient['Position'] = (-kR * float(aR) * rR / Dist**2 * (rR/Dist)**(aR-1.) +
                            2.*kA * D * S**2. / (D**2. + S**2.)**2.) * dot(Metric, Diff)/Dist * (-1) ** (wrtObject+1)
    return Gradient


def GradIntFunc(ObjVar, IntVar, ObjPar, IntPar):
    r0 = IntVar['r0']
    rR = IntVar['rR']
    S = IntVar['Sigma']
    aR = IntVar['aR']
    kA = IntVar['kA']
    kR = IntVar['kR']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))
    D = Dist - r0

    Gradient = dict()
    Gradient['r0'] = - 2.*kA * D * S**2. / (D**2. + S**2.)**2.
    Gradient['Sigma'] = - 2. * kA * D**2. * S / (D ** 2. + S ** 2.) ** 2.
    Gradient['kA'] = D**2. / (D**2. + S**2.)
    Gradient['rR'] = kR/Dist * aR * (rR/Dist)**(aR-1.)
    Gradient['aR'] = kR * (rR/Dist)**aR * log(aR)
    Gradient['kR'] = (rR/Dist)**aR
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

    kA = IntVar[0]
    r0 = IntVar[1]
    S = IntVar[2]
    kR = IntVar[3]
    rR = IntVar[4]
    aR = IntVar[5]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    dist = sqrt(dot(dot(IntPar[1:].reshape([dimI,dimI]), diff), diff))
    d = dist - r0

    energy = kR * (rR/dist)**aR + kA * d**2. / (d**2. + S**2.)
    return energy


def BareGradObj(ndarray ObjVar, ndarray IntVar, ndarray ObjPar, ndarray IntPar):
    """
    Bare form of object gradient function. Variables and parameters are ordered as:
    ObjVar = ([PosObj1], [PosObj2]) as one list
    IntVar = (RestLength, Stiffness)
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    :return: Energy and gradient w.r.t object variables
    """

    cdef int dimI = IntPar[0]

    cdef double kA = IntVar[0]
    cdef double r0 = IntVar[1]
    cdef double S = IntVar[2]
    cdef double kR = IntVar[3]
    cdef double rR = IntVar[4]
    cdef double aR = IntVar[5]

    cdef ndarray gradobj = ObjVar

    cdef double[:] diff = ObjVar[dimI:]
    cdef int i
    for i in range(dimI):
        diff[i] = diff[i] - ObjVar[i] # p2 - p1

    #cdef ndarray metricDot = dot(IntPar[1:].reshape([dimI, dimI]), diff)
    #cdef ndarray metricDot = dot(IntPar[1], diff)
    #cdef double dist = csqrt(dot(metricDot, diff))
    cdef double dist = 0.
    for i in range(dimI):
        dist += diff[i] ** 2
    dist = csqrt(dist) + 1.e-10

    cdef double d = dist - r0

    cdef double dps2 = d*d + S*S

    cdef double energy = 0
    energy += kR * (rR/dist) * (rR/dist)
    energy += kA * d*d / dps2

    cdef double dEdr = +kR * aR * rR / (dist * dist) * (rR / dist) ** (aR - 1.) - 2. * kA * d * S * S / (dps2 * dps2)

    for i in range(dimI):
        # gradobj[i] = (+kR * aR * rR / (dist*dist) * (rR/dist)**(aR-1.) -
        #                         2.*kA * d * S*S / (dps2*dps2)) * metricDot[i]/dist
        gradobj[i] = dEdr * diff[i] / dist
        gradobj[dimI+i] = - gradobj[i]

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

    kA = IntVar[0]
    r0 = IntVar[1]
    S = IntVar[2]
    kR = IntVar[3]
    rR = IntVar[4]
    aR = IntVar[5]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    #metricDot = dot(IntPar[1:].reshape([dimI, dimI]), diff)
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))
    d = dist - r0

    energy = kR * (rR / dist) ** aR + kA * d ** 2. / (d ** 2. + S ** 2.)

    gradint = empty_like(IntVar)
    gradint[0] = d**2. / (d**2. + S**2.)
    gradint[1] = - 2. * kA * d * S**2. / (d ** 2. + S ** 2.) ** 2.
    gradint[2] = - 2. * kA * d**2. * S / (d ** 2. + S ** 2.) ** 2.
    gradint[3] = (rR/dist)**aR
    gradint[4] = kR/dist * aR * (rR/dist)**(aR-1.)
    gradint[5] = kR * (rR/dist)**aR * log(aR)
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

    kA = IntVar[0]
    r0 = IntVar[1]
    S = IntVar[2]
    kR = IntVar[3]
    rR = IntVar[4]
    aR = IntVar[5]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    dist = sqrt(dot(dot(IntPar[1:].reshape([dimI,dimI]), diff), diff))
    d = dist - r0
    energy = kR * (rR / dist) ** aR + kA * d ** 2. / (d ** 2. + S ** 2.)
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
    :param params: A list of interaction parameters for all interactions [kA,Sigma,kR,rR,aR]
    :return I, A: Interaction List, Adjacency Matrix
    """

    Dim = Objs[0].Parameters['SpatialDimension']
    lo = len(Objs)

    for i in range(lo):
        assert Objs[i].Type == 'Mechanical Node'
        assert Objs[i].Parameters['SpatialDimension'] == Dim

    if not len(params):     # default values
        params = [1., 0.1, 0., 0., 12.]

    if type(Metric) == str:
        if Metric == 'Euclidean':
            Metric = eye(Dim)
    if shape(Metric) != (Dim, Dim):
        Metric = eye(Dim)  # if metric is not in right shape force Euclidean

    Interactions = []
    Adjacency = []

    kA = params[0]
    S = params[1]
    kR = params[2]
    rR = params[3]
    aR = params[4]

    for i in range(lo):
        P1 = Objs[i].Variables['Position']
        for j in range(i+1,lo):
            P2 = Objs[j].Variables['Position']
            Diff = P2 - P1
            Dist = sqrt(dot(dot(Metric, Diff), Diff))

            if Dist <= maxDist :  # Connect the two objects
                Interactions.append( makeInteraction(Dim, kA, Dist, S, kR, rR, aR, Metric) )
                A = zeros(lo)
                A[i] = 1
                A[j] = 1
                Adjacency.append(A)

    Adjacency = array(Adjacency)
    return Interactions, Adjacency