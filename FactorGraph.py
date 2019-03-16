from numpy import zeros, ravel, ndarray, size, array, zeros_like, meshgrid, float64, around, round, any, array_equal, r_, sqrt, inf, reshape, mean, argsort
from numpy.linalg import norm
from numpy.random import randn, normal
from matplotlib.mlab import find
from itertools import chain, combinations
from scipy.optimize import minimize, basinhopping
from collections import defaultdict
from copy import deepcopy
from os import getcwd
from sys import path as syspath
from importlib import import_module

from numpy cimport ndarray
cimport cython


syspath.append(getcwd()+'/MemoryModule/')
ObjectsModule = import_module('Objects')


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(True)

class FactorGraph:
    """
    A network of objects and interactions.
    """

    def __init__(self):
        """
        Initialize an empty factor graph.
        """

        self.Objects = dict()
        self.Interactions = dict()
        self.Adjacency = dict()
        return

    def ConnectNetwork(self, Objects, Interactions, Adjacency, CheckFlag = False):
        """
        Establish a connected network for one specific interaction.
        :param Objects:
        :param Interactions:
        :param Adjacency:
        """

        ObjType = Objects[0].Type
        self.Objects[ObjType] = Objects
        ObjectsModule._validateObjectCollection(self.Objects[ObjType])  # check object collection for proper initialization


        IntType = Interactions[0].Type
        self.Interactions[IntType] = Interactions

        self.Adjacency[IntType] = Adjacency

        def forids(id, i, Objects, Interactions):
            self.Interactions[IntType][i].Objects.append(Objects[id])
            cdef int cnt = len(self.Interactions[IntType][i].Objects) - 1
            if self.Objects[ObjType][id].Interactions == []:
                self.Objects[ObjType][id].Interactions = [[Interactions[i], cnt]]
            else:
                self.Objects[ObjType][id].Interactions.append([Interactions[i], cnt])
            return

        nI = len(Interactions)
        cdef int i
        for i in range(nI):
            self.Interactions[IntType][i].Objects = []
            ids = find(Adjacency[i])

            [forids(id,i,Objects,Interactions) for id in ids]

            self.Interactions[IntType][i]._validateInteraction(CheckFlag)
        return


    def getEnergy(self):
        """
        Get total energy from the network
        :return: Energy of all interactions
        """

        Keys = list(self.Interactions.keys())
        Ek = zeros(len(Keys))
        cdef int k
        for k in range(len(Keys)):
            Ints = self.Interactions[Keys[k]]
            Ek[k] = sum([ I.getEnergy() for I in Ints if I.Valid ])
        self.Energy = sum(Ek)
        return

    def getEnergyInteraction(self):
        """
        Get total energy from the network
        :return: Energy of all interactions
        """

        Keys = list(self.Interactions.keys())
        cdef int k
        E = dict()
        for k in range(len(Keys)):
            Ints = self.Interactions[Keys[k]]
            Ek = array([I.getEnergy() for I in Ints if I.Valid])
            E[k] = Ek
        self.EnergyInteraction = Ek
        return


    def getForceInteraction(self):
        """
        Get total energy from the network
        :return: Energy of all interactions
        """

        Keys = list(self.Interactions.keys())
        cdef int k
        F = dict()
        for k in range(len(Keys)):
            Ints = self.Interactions[Keys[k]]
            Fk = array([norm(I.getGradientObject(0)['Position']) for I in Ints if I.Valid])
            F[k] = Fk
        self.ForceInteraction = Fk
        return


    def getGradientObjects(self):
        """
        Get gradient of energy with respect to all object features
        :return: Gradient for each object due to all interactions it takes part in
        """

        Keys = list(self.Objects.keys())
        ObjGrad = dict()
        for k in Keys:
            Objs = self.Objects[k]
            OGrad = [[ Oi[0].getGradientObject(Oi[1]) for Oi in O.Interactions if Oi[0].Valid ] for O in Objs]
            ObjTotGrad = [ dsum(O) for O in OGrad ]
            ObjGrad[k] = ObjTotGrad
        self.GradientObjects = ObjGrad
        return


    def _makeGeneralizedDataStructures(self):
        """
        Introduce the generalized data structures used in bare energies and gradients
        """
        OneTypes = [int, float, float64]
        self._GVarInts = []
        self._GParInts = []
        self._GVarIntSlices = []
        self._GParIntSlices = []
        self._GTypeInt = []
        self._GAsocObjects = []
        self._GEnergies = []
        self._GGradObjs = []
        self._GGradInts = []
        self._GHessObjs = []
        self._GHessMixs = []

        self._GVarObjs = []
        self._GParObjs = []
        self._GVarObjSlices = []
        self._GParObjSlices = []

        self._GOrderedVarObj = []
        self._GOrderedVarObjInds = []
        self._GOrderedParObj = []
        self._GOrderedParObjInds = []
        self._GOrderedVarInt = []
        self._GOrderedVarIntInds = []
        self._GOrderedParInt = []
        self._GOrderedParIntInds = []

        self._GIntRecover = []
        self._GObjRecover = []
        self._GFixOrder = []

        self._GObjBounds = []
        self._GObjBoundsFlag = False
        self._GIntBounds = []
        self._GIntBoundsFlag = False

        self._TempObjects = []
        self._TempObjConstraint = dict()

        self._slcCntVar = 0
        self._slcCntPar = 0
        self._slcCntObjVar = 0
        self._slcCntObjPar = 0
        self._CntType = 0
        self._CntObj = 0
        self._CntInt = 0

        def forIVK(k, Int):
            if type(Int.Variables[k]) in OneTypes:
                self._GVarInts.append(Int.Variables[k])
                self._slcCntVar += 1
            if type(Int.Variables[k]) == list:
                self._GVarInts.extend(Int.Variables[k])
                self._slcCntVar += len(Int.Variables[k])
            if type(Int.Variables[k]) == ndarray:
                self._GVarInts.extend(ravel(Int.Variables[k]))
                self._slcCntVar += size(Int.Variables[k])

            if Int.Bounds:
                self._GIntBounds.append(Int.Bounds[k])
            return

        def forIPK(k, Int):
            if type(Int.Parameters[k]) in OneTypes:
                self._GParInts.append(Int.Parameters[k])
                self._slcCntPar += 1
            if type(Int.Parameters[k]) == list:
                self._GParInts.extend(Ints.Parameters[k])
                self._slcCntPar += len(Int.Parameters[k])
            if type(Int.Parameters[k]) == ndarray:
                self._GParInts.extend(ravel(Int.Parameters[k]))
                #self._GParInts.append(Int.Parameters[k])
                self._slcCntPar += size(Int.Parameters[k])
            return

        def forOVK(k, Obj):
            cdef int bi
            slc = self._slcCntObjVar
            if type(Obj.Variables[k]) in OneTypes:
                self._GVarObjs.append(Obj.Variables[k])
                self._slcCntObjVar += 1
                if Obj.Bounds == None:
                    self._GObjBounds.append([None, None])
                else:
                    self._GObjBoundsFlag = True
                    self._GObjBounds.append(Obj.Bounds[k])
            if type(Obj.Variables[k]) == list:
                ll = len(Obj.Variables[k])
                self._GVarObjs.extend(Obj.Variables[k])
                self._slcCntObjVar += ll
                if Obj.Bounds == None:
                    for bi in range(ll):
                        self._GObjBounds.append([None, None])
                else:
                    self._GObjBoundsFlag = True
                    for bi in range(ll):
                        self._GObjBounds.append(Obj.Bounds[k][bi])
            if type(Obj.Variables[k]) == ndarray:
                ll = size(Obj.Variables[k])
                self._GVarObjs.extend(ravel(Obj.Variables[k]))
                self._slcCntObjVar += ll
                if Obj.Bounds == None:
                    for bi in range(ll):
                        self._GObjBounds.append([None, None])
                else:
                    self._GObjBoundsFlag = True
                    for bi in range(ll):
                        self._GObjBounds.append(Obj.Bounds[k][bi])

            if Obj.GlobalConstraints != None:
                if k in self._TempObjConstraint[Obj.Type][Kind]['Keys']:
                    self._TempObjConstraint[Obj.Type][Kind]['Inds'].extend(list(range(slc, self._slcCntObjVar)))

            return

        def forOPK(k, Obj):
            if type(Obj.Parameters[k]) in OneTypes:
                self._GParObjs.append(Obj.Parameters[k])
                self._slcCntObjPar += 1
            if type(Obj.Parameters[k]) == list:
                self._GParObjs.extend(Obj.Parameters[k])
                self._slcCntObjPar += len(Obj.Parameters[k])
            if type(Obj.Parameters[k]) == ndarray:
                self._GParObjs.extend(ravel(Obj.Parameters[k]))
                #self._GParObjs.append(Obj.Parameters[k])
                self._slcCntObjPar += size(Obj.Parameters[k])
            return

        def forConstraint(Kind, OT, GC):
            if Kind not in self._TempObjConstraint[OT].keys():
                c = GC['Kinds'].index(Kind)
                self._TempObjConstraint[OT][Kind] = dict()
                self._TempObjConstraint[OT][Kind]['Keys'] = GC['Keys'][c]
                self._TempObjConstraint[OT][Kind]['Parameters'] = GC['Parameters'][c]
                self._TempObjConstraint[OT][Kind]['Inds'] = []
            return


        def forObjects(Obj):
            if Obj not in self._TempObjects:
                VarObjSlice = [self._slcCntObjVar]
                ParObjSlice = [self._slcCntObjPar]

                self._TempObjects.append(Obj)

                if Obj.GlobalConstraints != None:
                    OT = Obj.Type
                    if OT not in self._TempObjConstraint.keys():
                        self._TempObjConstraint[OT] = dict()

                    GC = Obj.GlobalConstraints
                    [forConstraint(Kind, OT, GC) for Kind in GC['Kinds']]

                self._GObjRecover.append([ Obj, self._CntObj ])
                self._GFixOrder.append(self.Objects[Obj.Type].index(Obj))

                OVK = Obj.VarKeys
                [forOVK(k, Obj) for k in OVK]

                VarObjSlice.append(self._slcCntObjVar)
                self._GVarObjSlices.append(VarObjSlice)
                self._ObjVarInds.extend(list(range(VarObjSlice[0], VarObjSlice[1])))

                OPK = Obj.ParKeys
                [forOPK(k, Obj) for k in OPK]

                ParObjSlice.append(self._slcCntObjPar)
                self._GParObjSlices.append(ParObjSlice)
                self._ObjParInds.extend(list(range(ParObjSlice[0], ParObjSlice[1])))

                self._AsocObjects.append(self._CntObj)
                self._CntObj += 1

            else:
                oid = self._TempObjects.index(Obj)
                self._AsocObjects.append(oid)
                self._ObjVarInds.extend(list(range(self._GVarObjSlices[oid][0], self._GVarObjSlices[oid][1])))
                self._ObjParInds.extend(list(range(self._GParObjSlices[oid][0], self._GParObjSlices[oid][1])))
            return


        AIK = list(self.Interactions.keys())
        cdef int ik
        for ik in range(len(AIK)):
            Ints = self.Interactions[AIK[ik]]

            for Int in Ints:
                if not Int.Valid:
                    continue

                self._GIntRecover.append([Int, self._CntInt])
                self._CntInt += 1

                self._GEnergies.append(Int.BareEnergy)
                self._GGradObjs.append(Int.BareGardObj)
                self._GGradInts.append(Int.BareGardInt)
                self._GHessObjs.append(Int.BareHessObj)
                self._GHessMixs.append(Int.BareHessMix)

                IntVarInds = []
                IntParInds = []

                VarSlice = [self._slcCntVar]
                ParSlice = [self._slcCntPar]
                self._GTypeInt.append(self._CntType)

                VK = Int.VarKeys
                [forIVK(k,Int) for k in VK]

                VarSlice.append(self._slcCntVar)
                self._GVarIntSlices.append(VarSlice)
                IntVarInds.extend(list(range(VarSlice[0], VarSlice[1])))

                PK = Int.ParKeys
                [forIPK(k,Int) for k in PK]

                ParSlice.append(self._slcCntPar)
                self._GParIntSlices.append(ParSlice)
                IntParInds.extend(list(range(ParSlice[0], ParSlice[1])))

                self._AsocObjects = []
                self._ObjVarInds = []
                self._ObjParInds = []
                [forObjects(Obj) for Obj in Int.Objects]

                self._GAsocObjects.append(self._AsocObjects)

                #TVO = array(self._GVarObjs)
                #TPO = array(self._GParObjs)
                self._GOrderedVarObj.append(array([ self._GVarObjs[idx] for idx in self._ObjVarInds ]))
                self._GOrderedVarObjInds.append(self._ObjVarInds)
                self._GOrderedParObj.append(array([ self._GParObjs[idx] for idx in self._ObjParInds ]))
                self._GOrderedParObjInds.append(self._ObjParInds)

                #TVI = array(self._GVarInts)
                #TPI = array(self._GParInts)
                self._GOrderedVarInt.append(array([ self._GVarInts[idx] for idx in IntVarInds ]))
                self._GOrderedVarIntInds.append(IntVarInds)
                self._GOrderedParInt.append(array([ self._GParInts[idx] for idx in IntParInds ]))
                self._GOrderedParIntInds.append(IntParInds)

            self._CntType += 1

        self._GVarInts = array(self._GVarInts)
        self._GVarObjs = array(self._GVarObjs)
        self._GParInts = (self._GParInts)
        self._GParObjs = (self._GParObjs)

        self._GFixOrder = argsort(self._GFixOrder)

        self._numIntList = list(range(len(self._GTypeInt)))
        self._TotalEnergy = self._makeGeneralizedEnergy()
        self._TotalEnergyInteraction = self._makeGeneralizedEnergyInteraction()
        self._GradObj = self._makeGeneralizedGradObj()
        self._GradInt = self._makeGeneralizedGradInt()
        self._HessObj = self._makeGeneralizedHessObj()
        self._HessMix = self._makeGeneralizedHessMix()

        self._GObjConstraints = []
        if self._TempObjConstraint :
            for Type in self._TempObjConstraint.keys() :
                for Kind in self._TempObjConstraint[Type].keys() :
                    FuncString = '_make_' + Kind
                    makeConstraint = getattr(ObjectsModule, FuncString)
                    self._GObjConstraints.append(makeConstraint(self._TempObjConstraint[Type][Kind]['Parameters'],
                                                                self._TempObjConstraint[Type][Kind]['Inds']))

        return

    def _makeGeneralizedEnergy(self):
        """
        Introduces a "fast" function to compute system energy with arrays and bare forms
        :return: Energy computation function
        """

        def TotalGeneralizedEnergy(VarObjs):
            energy = sum([
                self._GEnergies[I](
                    VarObjs[self._GOrderedVarObjInds[I]],
                    self._GOrderedVarInt[I],
                    self._GOrderedParObj[I],
                    self._GOrderedParInt[I],
                )
                for I in self._numIntList ]
            )
            return energy

        return TotalGeneralizedEnergy

    def _makeGeneralizedEnergyInteraction(self):
        """
        Introduces a "fast" function to compute system energy for all interactions with arrays and bare forms
        :return: Energy computation function
        """

        def TotalGeneralizedEnergyInteraction(VarObjs):
            energies = array([
                self._GEnergies[I](
                    VarObjs[self._GOrderedVarObjInds[I]],
                    self._GOrderedVarInt[I],
                    self._GOrderedParObj[I],
                    self._GOrderedParInt[I],
                )
                for I in self._numIntList ]
            )
            return energies

        return TotalGeneralizedEnergyInteraction

    def _makeGeneralizedGradObj(self):
        """
        Introduces a "fast" function to compute object gradients with arrays and bare forms
        :return: Energy & object gradient computation function
        """

        def TotalGeneralizedGradObj(ndarray VarObj, double RegLambda=0):
            # Can include a regularization term
            cdef double energy = 0.5 * RegLambda * sum(VarObj**2)
            cdef ndarray gradobj = RegLambda * VarObj

            cdef int I
            cdef int J

            cdef ndarray relevantVar
            cdef double Ei = 0.
            cdef ndarray gI = 0. * VarObj[self._GOrderedVarObjInds[0]]

            for I in range(len(self._numIntList)):
                relevantVar = VarObj[self._GOrderedVarObjInds[I]]
                eI, gI = self._GGradObjs[I](
                                                relevantVar,
                                                self._GOrderedVarInt[I],
                                                self._GOrderedParObj[I],
                                                self._GOrderedParInt[I]
                                                )
                energy += eI
                for J in range(len(self._GOrderedVarObjInds[I])):
                    gradobj[self._GOrderedVarObjInds[I][J]] += gI[J]

            return energy, gradobj

        return TotalGeneralizedGradObj


    def _makeGeneralizedGradInt(self):
        """
        Introduces a "fast" function to compute interaction gradients with arrays and bare forms
        :return: Energy & Interaction gradient computation function
        """

        def TotalGeneralizedGradInt(VarInt, RegLambda=0):
            # Can include a regularization term
            energy = 0. + 0.5 * RegLambda * sum(VarInt ** 2)
            gradint = zeros_like(VarInt) + RegLambda * VarInt
            for I in self._numIntList :
                relevantVar = VarInt[self._GOrderedVarIntInds[I]]
                eI, gI = self._GGradInts[I](
                                            self._GOrderedVarObj[I],
                                            relevantVar,
                                            self._GOrderedParObj[I],
                                            self._GOrderedParInt[I]
                                            )
                energy += eI
                gradint[self._GOrderedVarIntInds[I]] += gI

            return energy, gradint

        return TotalGeneralizedGradInt


    def _makeGeneralizedHessObj(self):
        """
        Introduces a "fast" function to compute derivatives
        :return: Object Hessian
        """

        def TotalGeneralizedHessObj(VarObj):
            cdef int I
            Hessobj = zeros([len(VarObj), len(VarObj)])
            for I in range(len(self._numIntList)):
                relevantVar = VarObj[self._GOrderedVarObjInds[I]]
                HI = self._GHessObjs[I](
                                            relevantVar,
                                            self._GOrderedVarInt[I],
                                            self._GOrderedParObj[I],
                                            self._GOrderedParInt[I]
                                            )
                rows, cols = meshgrid(self._GOrderedVarObjInds[I] , self._GOrderedVarObjInds[I])
                Hessobj[rows , cols] += HI

            return Hessobj

        return TotalGeneralizedHessObj

    def _makeGeneralizedHessMix(self):
        """
        Introduces a "fast" function to compute derivatives
        :return: "Mixed Hessian"
        """

        def TotalGeneralizedHessMix(VarAll):
            cdef int I
            HessMix = zeros([len(self._GVarInts), len(self._GVarObjs)])
            sep = len(self._GVarInts)
            for I in range(len(self._numIntList)):
                relevantVarInt = VarAll[self._GOrderedVarIntInds[I]]
                idxs = [idx+sep for idx in self._GOrderedVarObjInds[I]]
                relevantVarObj = VarAll[idxs]
                relevantVar = r_[relevantVarInt, relevantVarObj]
                HI = self._GHessMixs[I](
                                            relevantVar,
                                            self._GOrderedParObj[I],
                                            self._GOrderedParInt[I]
                                            )
                rows, cols = meshgrid(self._GOrderedVarIntInds[I] , self._GOrderedVarObjInds[I])
                HessMix[rows , cols] += HI.T

            return HessMix

        return TotalGeneralizedHessMix


    def OptimizeObjects(self, RegLambda = 0, Track = False, TrackEnergies = False, tol = None, step = None, GradientDescent = False, Mask = None, Sigma=None):
        """
        Find a local minimum by modifying object variables
        :return:
        """
        self._trackOptimization = []
        self._trackOptimizationEnergies = []

        if not GradientDescent:
            Args = dict()
            Args['fun'] = self._GradObj
            Args['x0']  = self._GVarObjs
            Args['jac'] = True
            if self._GObjConstraints :
                Args['constraints'] = self._GObjConstraints
            if self._GObjBoundsFlag :
                Args['bounds'] = self._GObjBounds
            if Track :
                Args['callback'] = self._Optimization_register
            Args['tol'] = tol

            self._ObjectOptimizationResult = minimize(**Args)
            self._GVarObjs = self._ObjectOptimizationResult.x
            self._StateEnergy = self._ObjectOptimizationResult.fun
        else:
            if step == None:
                step = 1.e-4
            #if tol == None:
            #    tol = 1.e-4
            if Mask == None:
                Mask = []
            x0 = self._GVarObjs
            X, E, Time = self._ObjectGradientDescent(x0, step, Mask, Track, TrackEnergies, Sigma)
            self._GVarObjs = X
            self._StateEnergy = E
            self._ConvergenceTime = Time

        NewVarObj = deepcopy(self._GVarObjs)
        PNVarObj = [NewVarObj[slice(self._GVarObjSlices[obj][0], self._GVarObjSlices[obj][1])]
                    for obj in list(range(len(self._GVarObjSlices)))]
        [ r[0]._reconstructObject(PNVarObj[r[1]]) for r in self._GObjRecover ]
        return

    def _Optimization_register(self, x):
        rx = around(x, 3)
        self._trackOptimization.append(rx)
        return


    def OptimizeObjectsTemperature(self, T=1.e-9, Iter=10):
        """
        Find a minimum by modifying object variables (with temperature)
        :return:
        """
        self._AllMinima = []
        min_kwargs = dict()
        min_kwargs['jac'] = True
        if self._GObjConstraints:
            min_kwargs['constraints'] = self._GObjConstraints
        if self._GObjBoundsFlag:
            min_kwargs['bounds'] = self._GObjBounds

        R = basinhopping(func=self._GradObj,
                         x0=self._GVarObjs,
                         T=T,
                         minimizer_kwargs=min_kwargs,
                         callback=self._basin_hop_register,
                         niter=Iter)

        self._ObjectOptimizationResult = R
        self._GVarObjs = self._ObjectOptimizationResult.x
        NewVarObj = self._GVarObjs
        PNVarObj = [NewVarObj[slice(self._GVarObjSlices[obj][0], self._GVarObjSlices[obj][1])]
                    for obj in list(range(len(self._GVarObjSlices)))]
        [ r[0]._reconstructObject(PNVarObj[r[1]]) for r in self._GObjRecover ]
        return


    def OptimizeInteractions(self, RegLambda = 0.):
        """
        Find a local minimum by modifying interaction variables
        :return:
        """
        assert (type(RegLambda) == float) and (RegLambda >= 0.)
        R = minimize(self._GradInt, self._GVarInts, args = (RegLambda,), jac = True, bounds = self._GIntBounds)
        if not R.success:
            print('Optimization failed!')
            return
        print([R.fun, 0.5 * RegLambda * sum(R.x ** 2)])
        R.fun = R.fun - 0.5 * RegLambda * sum(R.x ** 2)
        self._InteractionOptimizationResult = R
        self._GVarInts = self._InteractionOptimizationResult.x
        NewVarInt = self._GVarInts
        PNVarInt = [NewVarInt[slice(self._GVarIntSlices[int][0], self._GVarIntSlices[int][1])]
                    for int in list(range(len(self._GVarIntSlices)))]
        [ r[0]._reconstructInteraction(PNVarInt[r[1]]) for r in self._GIntRecover ]
        return


    def _ObjectGradientDescent(self, ndarray x0, double step, Mask = [], Track=False, TrackEnergies=False, Sigma=None):
        """
        Optimize using vanilla gradient descent
        :param self:
        :param x0: initial condition (of objects)
        :param step: step size
        :param tol: tolerance
        :param Mask: elements to zero the force on (i.e. elements that should not change)
        :return:
        """
        cdef double moveNorm = 0.
        cdef ndarray X = x0.copy()
        cdef ndarray Force
        cdef double Energy
        cdef int I
        cdef int Time = 0
        #cdef double sIntNum = len(self._GTypeInt)**0.5

        Energy, Force = self._GradObj(x0)
        cdef double FNorm = norm(Force)
        if FNorm < 1.e-12:
            return X, Energy, 0.
        for I in Mask:
            Force[I] = 0.

        cdef double convergence = Energy
        cdef int convCount = 0

        cdef float DX = 0.
        if Sigma != None:
            SThresh = 2.*Sigma
        else:
            SThresh = inf

        #while FNorm/sIntNum > tol:
        while convCount < 3 and DX <= SThresh:
            Time += 1
            X -= Force/FNorm * step
            DX = mean(abs(X-x0))
            #print(DX/SThresh)
            if Track:
                rx = around(X, 4)
                self._trackOptimization.append(rx)
            if TrackEnergies:
                ex = around(self._TotalEnergyInteraction(X),4)
                self._trackOptimizationEnergies.append(ex)
            Energy, Force = self._GradObj(X)
            FNorm = norm(Force)
            if Energy < convergence:
                convergence = Energy
                #convCount = 0
            else:
                convCount += 1
            #print(FNorm, Energy)
            for I in Mask:
                Force[I] = 0.

        if Track:
            rx = around(X, 4)
            self._trackOptimization.append(rx)
        if TrackEnergies:
            ex = around(self._TotalEnergyInteraction(X), 4)
            self._trackOptimizationEnergies.append(ex)

        return X, Energy, Time


    def List_Minima(self, T = 1.e10):
        self._AllMinima = []

        min_kwargs = dict()
        min_kwargs['jac'] = True
        if self._GObjConstraints:
            min_kwargs['constraints'] = self._GObjConstraints
        if self._GObjBoundsFlag:
            min_kwargs['bounds'] = self._GObjBounds

        R = basinhopping(func = self._GradObj,
                         x0 = self._GVarObjs,
                         T = T,
                         minimizer_kwargs = min_kwargs,
                         callback = self._basin_hop_register)
        return

    def _basin_hop_register(self, x, f, accept):
        rx = around(x,2)
        fx = round(f,2)

        AE = [array_equal(self._AllMinima[i][0], rx) for i in range(len(self._AllMinima))]
        if not any(AE) :
            G2 = self._GradObj(x)[1]**2.
            NG = sqrt(sum(G2))/len(G2)
            if NG <= 1.e-0 : # some cutoff for large gradients on boundaries (or numerical mistakes)
                self._AllMinima.append([ rx, fx, NG, 1 ])
            #self._AllMinima.append([rx, fx, NG, 1])
        else :
            self._AllMinima[AE.index(True)][3] += 1

        return


    def _MDDerivative(self):
        dVarObj = - self._GradObj(self._GVarObjs)[1]
        return dVarObj

    def _MDTimestep(self, tstep = 1.e-4, T=0.):
        dW = sqrt(2 * T) * normal(loc = 0.0, scale=sqrt(tstep))
        self._GVarObjs += self._MDDerivative() * tstep  +  dW
        return

    def _MDRun(self, Time, tstep=1.e-2 , T = 0.):
        self._Trajectory = [around(self._GVarObjs,10)]
        cdef double t = 0.
        while t < Time:
            self._MDTimestep(tstep, T)
            self._Trajectory.append(around(self._GVarObjs,10))
            t += tstep
        return


    def getObjects(self):
        Res = [ [ [ o.Variables[var] for var in o.Variables.keys() ] for o in self.Objects[k] ] for k in self.Objects.keys() ]
        return Res


    def ExploreMemoryBasin(self, ndarray Mem0, double Sigma, int N, int d, double SFrac=1.e-2, direction='Force'):
        """
        Estimate the size of memory basin and it's barrier height
        :param self:
        :return: estimates for attractor size and barrier height
        """
        Directions = ['Force', 'Random']
        cdef ndarray X = deepcopy(self._GVarObjs)
        EAtMemory, FAtMemory = self._GradObj(X)

        if direction not in Directions:
            direction = 'Force'

        if direction == 'Force' :
            if norm(FAtMemory) < 1.e-16:
                FAtMemory = randn(len(X))
            FNorm = FAtMemory/norm(FAtMemory)

        if direction == 'Random' :
            FNorm = randn(d*N)
            FNorm = FNorm / norm(FNorm)

        cdef ndarray DX = FNorm * Sigma * SFrac * sqrt(len(X))
        cdef double DXN = Sigma * SFrac * sqrt(len(X))
        cdef double RDX = 0.

        cdef step = 5.e-2*Sigma*sqrt(len(X))
        cdef ndarray XYF = reshape(deepcopy(X), [N, d]).T
        cdef double DM = mean(sqrt(sum((XYF - Mem0) ** 2., 0)))
        while DM < 5.e-2:
            EN = self._TotalEnergy(deepcopy(X))
            X -= DX
            RDX += DXN
            self._GVarObjs = deepcopy(X)
            self.OptimizeObjects(GradientDescent=True, Mask=[0,1,3], step = step)
            XYF = reshape(self._GVarObjs, [N, d]).T
            DM = mean(sqrt(sum((XYF - Mem0) ** 2., 0)))
            print(around(DM,6),around(RDX/sqrt(len(X)),6))


        #EN = self._TotalEnergy(X)
        #print(RDX, EN - EO)
        #while (EN > EO or RDX < Sigma) and (RDX < 1.):
        #    EO = EN
        #    EMin = min(EMin, EO)
        #    X -= DX
        #    RDX += DXN
        #    EN = self._TotalEnergy(X)
        #    #print(RDX, EN - EO)

        #DE = EO - EMin
        #if RDX >= 1.:
        #    return inf, inf

        #if DE == 0.0:
        #    R = 0.
        #else:
        #    R = RDX/sqrt(len(X))

        #print(X)
        return RDX/sqrt(len(X)), EN - EAtMemory




def RandomAdjacency(Objects, Interactions, ObjPerInt):
    """
    Generates a random network in which each interaction is tied to the appropriate number of objects.
    :param Objects: List of objects in the network.
    :param Interactions: List of interactions in the network.
    :param ObjPerInt: Number of objects attached to each interaction
    :return: Adjacency matrix
    """

    from numpy.random import randint

    nO = len(Objects)
    nI = len(Interactions)
    opi = ObjPerInt

    assert nO > 0
    assert nI > 0
    assert nO >= opi

    Adjacency = zeros([nI,nO])

    for i in range(nI):
        cnt = 0
        while cnt < opi:
            id = randint(nO)
            if not Adjacency[i,id]:
                Adjacency[i, id] = 1
                cnt = cnt + 1

    return Adjacency


def FullyConnectedAdjacency(Objects, ObjPerInt, makeInt, params):
    """
    Generates a fully connected network in which an interaction is tied to any combination of objects.
    :param Objects: List of objects in the network.
    :param ObjPerInt: Number of objects attached to each interaction
    :param makeInt: makeInteraction function
    :param params:  List of parameters for makeInt
    :return: Adjacency matrix
    """

    opi = ObjPerInt
    nO = len(Objects)
    assert nO > 0
    assert nO >= opi
    nI = int(round(choose(nO, opi)))

    Interactions = []

    Comb = array(list(combinations(range(nO),opi)))
    Adjacency = zeros([nI,nO])

    for i in range(nI):
        Interactions.extend(makeInt(tuple(params)))
        Adjacency[i,Comb[i]] = 1

    return Interactions, Adjacency



def dsum(dicts):
    """
    Function that merges and sums dictionaries
    :param dicts:
    :return:
    """
    ret = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            ret[k] += v
    return dict(ret)


def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0