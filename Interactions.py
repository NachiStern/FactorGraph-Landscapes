#from pylab import *
from numpy import intersect1d
cimport cython

@cython.profile(True)

class Interaction:
    """
    Interactions connect objects and carry the energy.
    Each interaction type is associated with the required type of "incoming" objects and an energy functional
    producing a (bound from below) scalar quantity due to the features of the objects.
    :Type: unique string identifying the interaction type
    :Variables: dictionary of features defining the interaction configuration space
    :Parameters: dictionary of parameters that are held constant
    :VarKeys: keys to the Variables dictionary setting the order of variables
    :ParKeys: keys to the Parameters dictionary setting the order of parameters
    :InputObjects: list of types + numbers of objects the interaction expects to connect
    :EnergyFunction: class-like energy computation
    :BareEnergy: array-like energy computation
    """

    def __init__(self, Type, Variables, Parameters, VarKeys, ParKeys, InputObjects,
                 EnergyFunction, BareEnergy,
                 GradObj = None, GradInt = None,
                 BareGradObj = None, BareGradInt = None,
                 BareHessObj = None, BareHessInt = None,
                 BareHessMix = None,
                 Bounds = None):
        """
        Initialize an Interaction
        :param Type:  Sets the type of the object, and consequently which objects it connects.
        """

        self.Type = Type
        self.Variables = Variables        # Relevant variables that can be tweaked (e.g. spring length, stiffness)
        self.VarKeys = VarKeys            # Sets the order of keys
        self.Parameters = Parameters      # Relevant extra parameters that stay constant (e.g. spatial metric)
        self.ParKeys = ParKeys            # Sets the order of keys
        self.InputObjects = InputObjects  # Object types and numbers interaction connects
        self.Energy = EnergyFunction      # Energy function for the interaction (Must supply)
        self.BareEnergy = BareEnergy      # Bare energy form (Must supply)
        self.GradObj = GradObj            # Gradients energy wrt to an object
        self.GradInt = GradInt            # Gradients energy wrt to interaction parameters
        self.BareGardObj = BareGradObj    # Bare object gradient form
        self.BareGardInt = BareGradInt    # Bare interaction gradient form
        self.BareHessObj = BareHessObj    # Bare object Hessian form
        self.BareHessInt = BareHessInt    # Bare interaction Hessian form
        self.BareHessMix = BareHessMix    # Bare "mixed Hessian" form
        self.Bounds = Bounds              # Lower and upper bounds for each variable
        self.Objects = []                 # Objects interacting through these interactions
        self.Valid = 0                    # flag to indicate well initialized and connected interaction
        return

    def getEnergy(self):
        ObjVar = []
        ObjPar = []
        for object in self.Objects:
            ObjVar.append(object.Variables)
            ObjPar.append(object.Parameters)
        Energy = self.Energy(ObjVar, self.Variables, ObjPar, self.Parameters)
        return Energy


    def getGradientObject(self, wrtObject):
        ObjVar = []
        ObjPar = []
        for object in self.Objects:
            ObjVar.append(object.Variables)
            ObjPar.append(object.Parameters)
        Gradient = self.GradObj(ObjVar, self.Variables, ObjPar, self.Parameters, wrtObject)
        return Gradient


    def getGradientInteraction(self):
        ObjVar = []
        ObjPar = []
        for object in self.Objects:
            ObjVar.append(object.Variables)
            ObjPar.append(object.Parameters)
        Gradient = self.GradInt(ObjVar, self.Variables, ObjPar, self.Parameters)
        return Gradient


    def _validateInteraction(self, CheckFlag=False):
        if CheckFlag:
            if self.Objects == []:  # No objects attached
                st = str(self) + ' is not connected to any objects'
                print(st)
                return
            else:
                for ov in range(len(self.InputObjects)):
                    cnt = 0
                    options = self.InputObjects[ov][0]
                    num = self.InputObjects[ov][1]
                    for on in range(cnt, cnt + num) :
                        try:    # Out of objects too soon
                            obj = self.Objects[on]
                        except:
                            st = str(self) + ' is lacking an object connection in spot #' + str(on+1)
                            print(st)
                            return

                        if obj.Type not in options :    # Object of wrong type
                            st = 'Expected object of type in ' + str(options) + ' on spot #' + str(on+1) + ', but received object of type ' + str(obj.Type)
                            print(st)
                            return

                        if not obj.Valid :          # Object is invalid for some reason
                            st = 'Object of type ' + str(obj.Type) + ' on spot #' + str(on+1) + 'is invalid'
                            print(st)
                            return

                        # Verify internal parameters of objects and interaction
                        SharedPars = intersect1d(self.ParKeys, obj.ParKeys)
                        for s in SharedPars :
                            if obj.Parameters[s] != self.Parameters[s] :  # parameters are not compatible
                                st = 'Parameter ("' + s + '" = ' + str(obj.Parameters[s]) + \
                                     ') of object on spot #' + str(on+1) + ' does not match the interaction parameter ("' + \
                                     s + '" = ' + str(self.Parameters[s]) + ')'
                                print(st)
                                return

                    cnt += num

            if len(self.Objects) > cnt :    # Too many objects
                self.Objects = self.Objects[:cnt]
                st = 'Interactions got more objects than it expects. Truncating the trailing objects.'
                print (st)

        self.Valid = 1
        return


    def _reconstructInteraction(self, VarInt):
        cnt = 0
        for k in self.VarKeys :
            tp = type(self.Variables[k])
            if tp in [int, float]:
                self.Variables[k] = VarInt[cnt]
                cnt += 1
            else:
                sp = self.Variables[k].shape
                sz = self.Variables[k].size
                self.Variables[k] = VarInt[cnt:cnt+sz].reshape(sp)
                cnt = cnt + sz
        return