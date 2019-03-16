## General interpolation between quadratic and Lorentz cup potentails

from numpy import eye, sqrt, dot, shape, reshape, array, empty_like, zeros, log, ctypeslib, outer, empty
from numpy cimport ndarray, int
from libc.math cimport sqrt as csqrt
cimport cython

from numdifftools import Hessian
from ..Interactions import Interaction

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(True)


def makeInteraction(Dim,kA,r0,Sigma,xsi,Metric='Euclidean',SmallR=0):
    Type = 'Generalized Lorentzian Cup'
    Variables = dict()
    Parameters = dict()

    assert int(Dim) > 0
    Dim = int(Dim)
    Parameters['SpatialDimension'] = Dim

    assert (r0 > 0.)
    Variables['r0'] = r0

    assert (Sigma > 0)
    Variables['Sigma'] = Sigma

    assert kA > 0
    Variables['kA'] = kA

    assert (xsi >= 0.) and (xsi <= 5.)
    Variables['xsi'] = xsi

    VarKeys = ['kA', 'r0', 'Sigma', 'xsi']

    if type(Metric) == str:
        if Metric == 'Euclidean':
            Metric = eye(Dim)
    if shape(Metric) != (Dim, Dim):
        Metric = eye(Dim)  # if metric is not in right shape force Euclidean
    Parameters['Metric'] = Metric

    assert (int(SmallR) >= 0) and (int(SmallR) <= 2)
    SmallR = int(SmallR)
    Parameters['SmallR'] = SmallR

    ParKeys = ['SpatialDimension', 'Metric', 'SmallR']

    InputObjects = [[['Mechanical Node'], 2]]

    Int = Interaction(Type, Variables, Parameters, VarKeys, ParKeys, InputObjects,
                      EnergyFunc, BareEnergy,
                      GradObj = GradObjFunc, GradInt = GradIntFunc,
                      BareGradObj = BareGradObj, BareGradInt = BareGradInt,
                      BareHessObj = BareHessObjAnalytic, BareHessInt = BareHessInt,
                      BareHessMix = BareHessMixed)
    return Int


def EnergyFunc(ObjVar, IntVar, ObjPar, IntPar):
    r0 = IntVar['r0']
    S = IntVar['Sigma']
    kA = IntVar['kA']
    xsi = IntVar['xsi']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))
    u = (Dist - r0)/S

    SmallR = IntPar['SmallR']
    SmallFlag = Dist < r0
    if (SmallR == 1) and SmallFlag :   # buckling rubber band
        Energy = 0.
    elif (SmallR == 2) and SmallFlag :  # linear small distance response
        Energy = kA * (S*u)**2.
    else:
        Energy = kA * (S**xsi) * u ** 2. / (1. + u ** 2.) ** (1.-0.5*xsi)

    return Energy


def GradObjFunc(ObjVar, IntVar, ObjPar, IntPar, wrtObject):
    r0 = IntVar['r0']
    S = IntVar['Sigma']
    kA = IntVar['kA']
    xsi = IntVar['xsi']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    assert (wrtObject < len(ObjVar)) and (wrtObject >= 0)

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))
    u = (Dist - r0) / S

    dEdr = 2.*kA * (S**(xsi-1.)) * (u + 0.5*xsi* u**3.) / (1. + u**2.)**(2.-0.5*xsi)
    drdx = dot(Metric, Diff)/Dist * (-1) ** (wrtObject+1)

    SmallR = IntPar['SmallR']
    SmallFlag = Dist < r0
    Gradient = dict()
    if (SmallR == 1) and SmallFlag :   # buckling rubber band
        Gradient['Position'] = 0.
    elif (SmallR == 2) and SmallFlag :  # linear small distance response
        Gradient['Position'] = 2.*kA * S * u * drdx
    else:
        Gradient['Position'] = dEdr * drdx
    return Gradient


def GradIntFunc(ObjVar, IntVar, ObjPar, IntPar):
    r0 = IntVar['r0']
    S = IntVar['Sigma']
    kA = IntVar['kA']
    xsi = IntVar['xsi']
    Metric = IntPar['Metric']

    PosObj1 = ObjVar[0]['Position']
    PosObj2 = ObjVar[1]['Position']

    Diff = PosObj2 - PosObj1
    Dist = sqrt(dot(dot(Metric, Diff), Diff))
    u = (Dist - r0) / S

    dEdu = 2. * kA * (S**xsi) * (u + 0.5*xsi * u ** 3.) / (1. + u ** 2.) ** (2.-0.5*xsi)

    SmallR = IntPar['SmallR']
    SmallFlag = Dist < r0
    Gradient = dict()
    if (SmallR == 1) and SmallFlag :   # buckling rubber band
        Gradient['r0'] = 0.
        Gradient['Sigma'] = 0.
        Gradient['kA'] = 0.
        Gradient['xsi'] = 0.
    elif (SmallR == 2) and SmallFlag :  # linear small distance response
        Gradient['r0'] = - 2. * kA * u / S
        Gradient['Sigma'] = - 2. * kA * u * u / S
        Gradient['kA'] = u ** 2.
        Gradient['xsi'] = 0.
    else:
        Gradient['r0'] = - dEdu / S
        Gradient['Sigma'] = - dEdu * u / S
        Gradient['kA'] = u ** 2. / (1. + u ** 2.) ** (1.-0.5*xsi)
        Gradient['xsi'] = - kA * log(1. + u ** 2.) * u ** 2. / (1. + u ** 2.) ** (1.-0.5*xsi)

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
    xsi = IntVar[3]


    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    dist = sqrt(dot(dot(IntPar[1:1+dimI*dimI].reshape([dimI,dimI]), diff), diff))
    u = (dist - r0) / S

    SmallR = IntPar[1+dimI*dimI]
    SmallFlag = dist < r0
    if (SmallR == 1) and SmallFlag :   # buckling rubber band
        energy = 0.
    elif (SmallR == 2) and SmallFlag :  # linear small distance response
        energy = kA * (S*u) ** 2.
    else:
        energy = kA * (S**xsi) * u ** 2. / (1. + u ** 2.) ** (1.-0.5*xsi)

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
    cdef double xsi = IntVar[3]

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
    dist = csqrt(dist) #+ 1.e-16

    cdef double[:] drdx = empty(dimI)
    if dist > 1.e-16:
        for i in range(dimI):
            drdx[i] = diff[i]/dist
    else:
        for i in range(dimI):
            drdx[i] = 1.

    cdef double u = (dist - r0) / S
    cdef double u2 = u * u

    cdef double energy

    cdef double dEdu = - 2. * kA * (S**xsi) * (u + 0.5*xsi*u**3.) / (1. + u2)**(2.-0.5*xsi)
    cdef double dEdr = dEdu / S

    cdef int SmallR = IntPar[1+dimI*dimI]
    SmallFlag = dist < r0
    if (SmallR == 1) and SmallFlag:  # buckling rubber band
        energy = 0.
        for i in range(dimI):
            gradobj[i] = 0.
            gradobj[dimI+i] = 0.
    elif (SmallR == 2) and SmallFlag:  # linear small distance response
        energy = kA * (S*u) ** 2.
        for i in range(dimI):
            gradobj[i] = - 2. * kA * (S*u) * drdx[i]
            gradobj[dimI+i] = - gradobj[i]
    else:
        energy = kA * (S**xsi) * u ** 2. / (1. + u ** 2.) ** (1.-0.5*xsi)
        for i in range(dimI):
            #gradobj[i] = dEdr * diff[i] / dist
            gradobj[i] = dEdr * drdx[i]
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
    xsi = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    #metricDot = dot(IntPar[1:].reshape([dimI, dimI]), diff)
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))
    u = (dist - r0) / S

    energy = kA * u**2. / (1. + u**2.)**xsi

    dEdu = - 2. * kA * (S**xsi) * (u + 0.5*xsi * u ** 3.) / (1. + u ** 2.) ** (2.-0.5*xsi)

    gradint = empty_like(IntVar)
    cdef int SmallR = IntPar[1 + dimI * dimI]
    SmallFlag = dist < r0

    if (SmallR == 1) and SmallFlag:  # buckling rubber band
        energy = 0.
        gradint[0] = 0.
        gradint[1] = 0.
        gradint[2] = 0.
        gradint[3] = 0.
    elif (SmallR == 2) and SmallFlag:  # linear small distance response
        energy = kA * u ** 2.
        gradint[0] = energy / kA
        gradint[1] = 2. * kA * u / S
        gradint[2] = 2. * kA * u * u / S
        gradint[3] = 0.
    else:
        energy = kA * u ** 2. / (1. + u ** 2.) ** (1-0.5*xsi)
        gradint[0] = energy / kA
        gradint[1] = - dEdu / S
        gradint[2] = - dEdu * u / S
        gradint[3] = - kA * log(1. + u ** 2.) * u ** 2. / (1. + u ** 2.) ** (1-0.5*xsi)

    gradint[0] = energy / kA
    gradint[1] = - dEdu / S
    gradint[2] = - dEdu * u / S
    gradint[3] = - kA * log(1. + u ** 2.) * u**2. / (1. + u**2.)**xsi
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


def BareHessObjAnalytic(ObjVar, IntVar, ObjPar, IntPar):
    """
        Bare form of object Hessian function. Variables and parameters are ordered as:
        ObjVar = ([PosObj1], [PosObj2]) as one list
        IntVar = (RestLength, Stiffness)
        ObjPar = (Spatial dimension 1 , Spatial dimension 2)
        IntPar = (Spatial dimension, [metric])
        :return: Hessian matrix w.r.t object variables
    """

    dimI = int(IntPar[0])

    kA = IntVar[0]
    r0 = IntVar[1]
    S = IntVar[2]
    xsi = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    # metricDot = dot(IntPar[1:].reshape([dimI, dimI]), diff)
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))
    u = (dist - r0) / S

    c1 = (1.-xsi)*u*u
    c2 = 1. + u*u

    C = ((1. + 3.*c1) * c2 - 2.*(2.-0.5*xsi)*u*u*(1. + c1)) / c2**(3.-0.5*xsi)
    d2edr2 = 2. * kA * C / (S*S)

    HessObj = zeros([len(ObjVar), len(ObjVar)])
    cdef int SmallR = IntPar[1 + dimI * dimI]
    SmallFlag = dist < r0

    if (SmallR == 1) and SmallFlag:  # buckling rubber band
        Hblock = zeros(dimI,dimI)
        HessObj[:dimI, :dimI] = Hblock
        HessObj[dimI:, dimI:] = Hblock
        HessObj[:dimI, dimI:] = - Hblock
        HessObj[dimI:, :dimI] = - Hblock
    elif (SmallR == 2) and SmallFlag:  # linear small distance response
        Hblock = 2. * kA / (S*S) * outer(metricDot, metricDot) / (dist * dist)
        HessObj[:dimI, :dimI] = Hblock
        HessObj[dimI:, dimI:] = Hblock
        HessObj[:dimI, dimI:] = - Hblock
        HessObj[dimI:, :dimI] = - Hblock
    else:
        Hblock = d2edr2 * outer(metricDot, metricDot) / (dist * dist)
        HessObj[:dimI, :dimI] = Hblock
        HessObj[dimI:, dimI:] = Hblock
        HessObj[:dimI, dimI:] = - Hblock
        HessObj[dimI:, :dimI] = - Hblock

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
    xsi = IntVar[3]

    diff = ObjVar[dimI:] - ObjVar[:dimI]  # p2 - p1
    dist = sqrt(dot(dot(IntPar[1:].reshape([dimI,dimI]), diff), diff))
    u = (dist - r0)/S

    SmallR = IntPar[1 + dimI * dimI]
    SmallFlag = dist < r0
    if (SmallR == 1) and SmallFlag:  # buckling rubber band
        energy = 0.
    elif (SmallR == 2) and SmallFlag:  # linear small distance response
        energy = kA * (S*u) ** 2.
    else:
        energy = kA * (S**xsi) * u ** 2. / (1. + u ** 2.) ** (1.-0.5*xsi)

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


def BareHessMixed(VarAll, ObjPar, IntPar):
    """
    Bare form of the "mixed" Hessian function; derivatives are taken with respect to both objects and interactions
    :param VarAll: first interaction variables and then object variables
    :param ObjPar:
    :param IntPar:
    :return:
    """

    dimI = int(IntPar[0])

    kA = VarAll[0]
    r0 = VarAll[1]
    S = VarAll[2]
    xsi = VarAll[3]

    diff = VarAll[4+dimI:] - VarAll[4:4+dimI]  # p2 - p1
    # metricDot = dot(IntPar[1:].reshape([dimI, dimI]), diff)
    metricDot = dot(IntPar[1], diff)
    dist = sqrt(dot(metricDot, diff))
    u = (dist - r0) / S

    R = metricDot / dist
    c1 = 0.5*xsi * u * u
    c2 = 1. + u * u
    B = u * (1. + c1) / c2**(xsi+1.)
    C = ((1. + 3. * c1) * c2 - 2. * (2.-0.5*xsi) * u * u * (1. + c1)) / c2 ** (3.-0.5*xsi)
    E = - u * ((1.-0.5*xsi)*u*u + c1*log(c2)) / c2**(2.-0.5*xsi)

    HessMix = zeros([4, 2*dimI])
    cdef int SmallR = IntPar[1 + dimI * dimI]
    SmallFlag = dist < r0

    if (SmallR == 1) and SmallFlag:  # buckling rubber band
        HessMix = zeros([4, 2*dimI])
    elif (SmallR == 2) and SmallFlag:  # linear small distance response
        HessMix[0, :dimI] = - 2. * u * R
        HessMix[0, dimI:] = - HessMix[0, :dimI]
        HessMix[1, :dimI] = (+ 2. * kA  * u / S) * R
        HessMix[1, dimI:] = - HessMix[1, :dimI]
        HessMix[2, :dimI] = (+ 2, * kA * u * u /S) * R
        HessMix[2, dimI:] = - HessMix[2, :dimI]
        HessMix[3, :dimI] = 0.
        HessMix[3, dimI:] = 0.
    else:
        HessMix[0, :dimI] = - 2. / S * B * R
        HessMix[0, dimI:] = - HessMix[0, :dimI]
        HessMix[1, :dimI] = + 2. * kA / (S * S) * C * R
        HessMix[1, dimI:] = - HessMix[1, :dimI]
        HessMix[2, :dimI] = + 2. * kA / (S * S) * (B + u * C) * R
        HessMix[2, dimI:] = - HessMix[2, :dimI]
        HessMix[3, :dimI] = + 2. * kA / S * E * R
        HessMix[3, dimI:] = - HessMix[3, :dimI]

    return HessMix


def ConnectStablyByDistance(Objs, maxDist, params = [], Metric = [], smallR=0):
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

    if not len(params):
        params = [1., 0.1, 1.]

    if type(Metric) == str:
        if Metric == 'Euclidean':
            Metric = eye(Dim)
    if shape(Metric) != (Dim, Dim):
        Metric = eye(Dim)  # if metric is not in right shape force Euclidean

    Interactions = []
    Adjacency = []

    kA = params[0]
    S = params[1]
    xsi = params[2]

    for i in range(lo):
        P1 = Objs[i].Variables['Position']
        for j in range(i+1,lo):
            P2 = Objs[j].Variables['Position']
            Diff = P2 - P1
            Dist = sqrt(dot(dot(Metric, Diff), Diff))

            if Dist <= maxDist :  # Connect the two objects
                Interactions.append( makeInteraction(Dim, kA, Dist, S, xsi, Metric, smallR) )
                A = zeros(lo)
                A[i] = 1
                A[j] = 1
                Adjacency.append(A)

    Adjacency = array(Adjacency)
    return Interactions, Adjacency