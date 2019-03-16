## Interaction of degree p muliplying spins with a linear coupling constant J

from numpy import eye, sqrt, array, zeros, prod
from ..Interactions import Interaction

def makeInteraction(p,J):
    """
    make a p-spin coupling interaction
    :param p: degree of coupling
    :param J: coupling constant
    :return:
    """
    assert int(p) > 0
    Type = str(int(p)) + '-Spin Coupling'
    Variables = dict()
    Parameters = dict()

    Parameters['p'] = int(p)

    assert type(J) == float
    Variables['J'] = J
    VarKeys = ['J']

    ParKeys = ['p']

    InputObjects = [[['Spherical Spin', 'Ising Spin'], p]]

    Int = Interaction(Type, Variables, Parameters, VarKeys, ParKeys, InputObjects,
                      EnergyFunc, BareEnergy,
                      GradObj = GradObjFunc, GradInt = GradIntFunc,
                      BareGradObj = BareGradObj, BareGradInt = BareGradInt,
                      BareHessObj = BareHessObj, BareHessInt = BareHessInt)
    return Int


def EnergyFunc(ObjVar, IntVar, ObjPar, IntPar):
    p = IntPar['p']
    J = IntVar['J']

    Spins = []
    for objvar in ObjVar:
        Spins.append(objvar['Value'])

    Energy = J * prod(Spins)
    return Energy


def GradObjFunc(ObjVar, IntVar, ObjPar, IntPar, wrtObject):
    p = IntPar['p']
    J = IntVar['J']

    assert (wrtObject < len(ObjVar)) and (wrtObject >= 0)

    Spins = []
    for objvar in ObjVar:
        Spins.append(objvar['Value'])

    Spins.pop(wrtObject)

    Gradient = dict()
    Gradient['Value'] = J * prod(Spins)
    return Gradient


def GradIntFunc(ObjVar, IntVar, ObjPar, IntPar):
    p = IntPar['p']
    J = IntVar['J']

    Spins = []
    for objvar in ObjVar:
        Spins.append(objvar['Value'])

    Gradient = dict()
    Gradient['J'] = prod(Spins)
    return Gradient


def BareEnergy(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of energy function. Variables and parameters are ordered as:
    ObjVar = ([PosObj1], [PosObj2]) as one list
    IntVar = (RestLength, Stiffness)
    ObjPar = (Spatial dimension 1 , Spatial dimension 2)
    IntPar = (Spatial dimension, [metric])
    """
    #p = IntPar[0]
    J = IntVar[0]

    energy = J * prod(ObjVar)
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
    p = IntPar[0]
    J = IntVar[0]

    energy = J * prod(ObjVar)

    pset = set(range(p))
    gradobj = J * array([ prod(ObjVar[array(list(pset - set([n])))]) for n in range(p) ])
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
    p = IntPar[0]
    J = IntVar[0]

    energy = J * prod(ObjVar)

    gradint = array([prod(ObjVar)])
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
    p = IntPar[0]
    J = IntVar[0]

    #energy = J * prod(ObjVar)

    pset = set(range(p))

    #HessObj = zeros([len(ObjVar), len(ObjVar)])
    HessObj = J * array([[ prod(ObjVar[array(list(pset - set([n]) - set([m])))]) for n in range(p) if n != m] for m in range(p) ])
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
    HessInt = array([0.])
    return HessInt


def Random_Couplings(N,p,mean,variance):
    from numpy.random import randn

    Interactions = []
    for i in range(N):
        J = randn() * sqrt(variance) + mean
        Interactions.append(makeInteraction(p,float(J)))

    return Interactions

def Random_Couplings2(args):
    from numpy.random import randn

    Interactions = []
    for i in range(args[0]):
        J = randn() * sqrt(args[3]) + args[2]
        Interactions.append(makeInteraction(args[1],float(J)))

    return Interactions