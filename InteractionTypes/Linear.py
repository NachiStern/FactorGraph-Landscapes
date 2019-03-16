"""
A linear interaction of type (Ax + b = 0) with (x) the objects and (A,b) parameters.
Note that each interaction is one "line" of the linear system, such that A is a vector and b is a scalar.
"""

from numpy import ndarray, dot, empty, outer


def makeInteraction(A, b, xdim):
    """
    make a linear coupling interaction
    :param A: linear coupling of objects
    :param b: bias
    :param xdim: number of scalar value objects (dim of variable vector x)
    :return:
    """
    assert type(A) == ndarray
    assert type(b) == float
    assert int(x) > 0
    Type = 'Linear'
    Variables = dict()
    Parameters = dict()

    Parameters['xdim'] = int(xdim)

    Variables['A'] = A
    Variables['b'] = b
    VarKeys = ['A','b']

    ParKeys = ['xdim']

    InputObjects = [[['ScalarValue'], xdim]]

    Int = Interaction(Type, Variables, Parameters, VarKeys, ParKeys, InputObjects,
                      EnergyFunc, BareEnergy,
                      GradObj = GradObjFunc, GradInt = GradIntFunc,
                      BareGradObj = BareGradObj, BareGradInt = BareGradInt,
                      BareHessObj = BareHessObj, BareHessInt = BareHessInt)
    return Int


def EnergyFunc(ObjVar, IntVar, ObjPar, IntPar):
    xdim = IntPar['xdim']
    A = IntVar['A']
    b = IntVar['b']

    Vec = []
    for objvar in ObjVar:
        Vec.append(objvar['Value'])

    Energy = 0.5 * (dot(A,Vec) + b)**2.
    return Energy


def GradObjFunc(ObjVar, IntVar, ObjPar, IntPar, wrtObject):
    xdim = IntPar['xdim']
    A = IntVar['A']
    b = IntVar['b']

    assert (wrtObject < len(ObjVar)) and (wrtObject >= 0)

    Vec = []
    for objvar in ObjVar:
        Vec.append(objvar['Value'])

    Gradient = dict()
    Gradient['Value'] = (dot(A,Vec) + b) * A[wrtObject]
    return Gradient


def GradIntFunc(ObjVar, IntVar, ObjPar, IntPar):
    xdim = IntPar['xdim']
    A = IntVar['A']
    b = IntVar['b']

    Vec = []
    for objvar in ObjVar:
        Vec.append(objvar['Value'])

    scProd = dot(A, Vec) + b

    Gradient = dict()
    Gradient['A'] = scProd * Vec
    Gradient['b'] = scProd
    return Gradient


def BareEnergy(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of energy function. Variables and parameters are ordered as:
    ObjVar = (Vector components) as one list
    IntVar = (b, [A])
    ObjPar = ()
    IntPar = (xdim)
    """
    scProd = intVar[0] + dot(IntVar[1:], ObjVar)

    energy = 0.5 * scProd**2.
    return energy


def BareGradObj(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object gradient function. Variables and parameters are ordered as:
    ObjVar = (Vector components) as one list
    IntVar = ([A], b)
    ObjPar = ()
    IntPar = (xdim)
    :return: Energy and gradient w.r.t object variables
    """
    scProd = intVar[0] + dot(IntVar[1:], ObjVar)

    energy = 0.5 * scProd ** 2.
    gradobj = scProd * IntVar[1:]
    return energy, gradobj


def BareGradInt(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object gradient function. Variables and parameters are ordered as:
    ObjVar = (Vector components) as one list
    IntVar = ([A], b)
    ObjPar = ()
    IntPar = (xdim)
    :return: Energy and gradient w.r.t interaction variables
    """
    scProd = intVar[0] + dot(IntVar[1:], ObjVar)

    energy = 0.5 * scProd ** 2.
    gradint = zeros(xdim + 1)
    gradint[0] = scProd
    gradint[1:] = scProd * ObjVar
    return energy, gradint


def BareHessObj(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object Hessian function. Variables and parameters are ordered as:
    ObjVar = (Vector components) as one list
    IntVar = ([A], b)
    ObjPar = ()
    IntPar = (xdim)
    :return: Hessian matrix w.r.t object variables
    """
    HessObj = outer(IntVar[1:], IntVar[1:])
    return HessObj


def BareHessInt(ObjVar, IntVar, ObjPar, IntPar):
    """
    Bare form of object Hessian function. Variables and parameters are ordered as:
    ObjVar = (Vector components) as one list
    IntVar = ([A], b)
    ObjPar = ()
    IntPar = (xdim)
    :return: Hessian matrix w.r.t object variables
    """
    xdim = IntPar[0]
    HessInt = empty(xdim + 1, xdim + 1)
    HessInt[0,0]   = 1.
    HessInt[1:,0]  = ObjVar
    HessInt[0,1:]  = ObjVar
    HessInt[1:,1:] = outer(ObjVar, ObjVar)
    return HessInt


def Random_Couplings(N,xdim,mean,variance):
    from numpy.random import randn

    Interactions = []
    for i in range(N):
        A = randn(xdim) * sqrt(variance) + mean
        b = randn() * sqrt(variance) + mean
        Interactions.append(makeInteraction(A, b, xdim))

    return Interactions