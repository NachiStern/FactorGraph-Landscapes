#from pylab import *
from numpy import zeros_like
from numpy.linalg import norm

class Object:
    """
    Objects interact via potentials to create energy landscape.
    Each object type has an associated set of features, and a set of interaction types through which it interacts
    with other objects.
    """

    def __init__(self, Type, Variables, Parameters, VarKeys, ParKeys, GlobalConstraints = None, Bounds = None):
        """
        Initialize an object.
        :param Type: Sets the type of the object, and consequently how it is allowed to interact.
        """

        self.Type = Type
        self.Variables = Variables    # Variables associated with object
        self.VarKeys = VarKeys        # Sets the order of keys
        self.Parameters = Parameters  # Extra parameters associated with object, that stay constant
        self.ParKeys = ParKeys        # Sets the order of keys
        self.Interactions = []        # Which interaction does the object take part in, and as which input
        self.GlobalConstraints = GlobalConstraints      # Constraints on all objects
        self.Bounds = Bounds          # Lower\upper bounds on object variables
        self.Valid = 0
        return


    def _reconstructObject(self, VarObj):
        cnt = 0
        for k in self.VarKeys :
            tp = type(self.Variables[k])
            if tp in [int, float]:
                self.Variables[k] = VarObj[cnt]
                cnt += 1
            else:
                sp = self.Variables[k].shape
                sz = self.Variables[k].size
                self.Variables[k] = VarObj[cnt:cnt+sz].reshape(sp)
                cnt = cnt + sz
        return


def _validateObjectCollection(Objects):
    for obj in range(len(Objects)) :
        GC = Objects[obj].GlobalConstraints
        if obj == 0 :
            GlobalConstraints = GC
        if (obj > 0) and (GC != GlobalConstraints) :  # inconsistent global constraints within object collection
            st = 'Global constraints of object #' + str(obj+1) + ': ' + str(GC) + ' are inconsistent with ' + str(GlobalConstraints)
            print(st)
            return

    for obj in Objects: # objects group is validated
        obj.Valid = 1
    return


### Global norm constraint
def _NormConstraint(ObjVar,Norm,inds):
    C = 0.5 * (norm(ObjVar[inds]) - Norm)**2
    return C

def _NormConstraint_Gradient(ObjVar,Norm,inds):
    G = zeros_like(ObjVar)
    N = norm(ObjVar[inds])
    G[inds] = ObjVar[inds] * (1. - Norm/N)
    #print([norm(ObjVar[inds]),G])
    return G

def _make_NormConstraint(params, inds):
    Norm = params[0]
    d = dict()
    d['type'] = 'eq'
    d['fun'] = _NormConstraint
    #d['jac'] = _NormConstraint_Gradient
    d['args'] = [Norm, inds]
    return d
