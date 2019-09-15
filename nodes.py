from collections import OrderedDict
from itertools import product

from sympy import Basic
from sympy.core.singleton import Singleton
from sympy.core.compatibility import with_metaclass
from sympy.core.containers import Tuple
from sympy import AtomicExpr

from sympde.topology import ScalarTestFunction, VectorTestFunction
from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import Mapping


#==============================================================================
class ArityType(with_metaclass(Singleton, Basic)):
    """Base class representing a form type: bilinear/linear/functional"""
    pass

class BilinearArity(ArityType):
    pass

class LinearArity(ArityType):
    pass

class FunctionalArity(ArityType):
    pass

#==============================================================================
class IndexNode(with_metaclass(Singleton, Basic)):
    """Base class representing one index of an iterator"""
    pass

class IndexElement(IndexNode):
    pass

class IndexQuadrature(IndexNode):
    pass

class IndexDof(IndexNode):
    pass

class IndexDofTrial(IndexNode):
    pass

class IndexDofTest(IndexNode):
    pass

class IndexDerivative(IndexNode):
    pass

index_element   = IndexElement()
index_quad      = IndexQuadrature()
index_dof       = IndexDof()
index_dof_trial = IndexDofTrial()
index_dof_test  = IndexDofTest()
index_deriv     = IndexDerivative()

#==============================================================================
class LengthNode(with_metaclass(Singleton, Basic)):
    """Base class representing one length of an iterator"""
    pass

class LengthElement(LengthNode):
    pass

class LengthQuadrature(LengthNode):
    pass

class LengthDof(LengthNode):
    pass

class LengthDofTrial(LengthNode):
    pass

class LengthDofTest(LengthNode):
    pass

length_element   = LengthElement()
length_quad      = LengthQuadrature()
length_dof       = LengthDof()
length_dof_trial = LengthDofTrial()
length_dof_test  = LengthDofTest()

#==============================================================================
class RankNode(with_metaclass(Singleton, Basic)):
    """Base class representing a rank of an iterator"""
    pass

class RankDimension(RankNode):
    pass

rank_dim = RankDimension()

#==============================================================================
class BaseNode(Basic):
    """
    """
    pass

#==============================================================================
class Element(BaseNode):
    """
    """
    pass

#==============================================================================
class Pattern(Tuple):
    """
    """
    pass

#==============================================================================
class IteratorBase(BaseNode):
    """
    """
    def __new__(cls, target, dummies=None):
        if not dummies is None:
            if not isinstance(dummies, (list, tuple, Tuple)):
                dummies = [dummies]
            dummies = Tuple(*dummies)

        return Basic.__new__(cls, target, dummies)

    @property
    def target(self):
        return self._args[0]

    @property
    def dummies(self):
        return self._args[1]

#==============================================================================
class TensorIterator(IteratorBase):
    pass

#==============================================================================
class ProductIterator(IteratorBase):
    pass

#==============================================================================
# TODO dummies should not be None
class GeneratorBase(BaseNode):
    """
    """
    def __new__(cls, target, dummies):
        if not isinstance(dummies, (list, tuple, Tuple)):
            dummies = [dummies]
        dummies = Tuple(*dummies)

        if not isinstance(target, (ArrayNode, MatrixNode)):
            raise TypeError('expecting an ArrayNode')

        return Basic.__new__(cls, target, dummies)

    @property
    def target(self):
        return self._args[0]

    @property
    def dummies(self):
        return self._args[1]

#==============================================================================
class TensorGenerator(GeneratorBase):
    pass

#==============================================================================
class ProductGenerator(GeneratorBase):
    pass

#==============================================================================
class Grid(BaseNode):
    """
    """
    pass

#==============================================================================
class ScalarNode(BaseNode, AtomicExpr):
    """
    """
    pass

#==============================================================================
class ArrayNode(BaseNode, AtomicExpr):
    """
    """
    _rank = None
    _positions = None
    _free_positions = None

    @property
    def rank(self):
        return self._rank

    @property
    def positions(self):
        return self._positions

    @property
    def free_positions(self):
        if self._free_positions is None:
            return list(self.positions.keys())

        else:
            return self._free_positions

    def pattern(self, args=None):
        if args is None:
            args = self.free_positions

        positions = {}
        for a in args:
            positions[a] = self.positions[a]

        args = [None]*self.rank
        for k,v in positions.items():
            args[v] = k

        return Pattern(*args)

#==============================================================================
class MatrixNode(BaseNode, AtomicExpr):
    """
    """
    _rank = None

    @property
    def rank(self):
        return self._rank

    def pattern(self, positions):
        raise NotImplementedError('TODO')

#==============================================================================
class GlobalTensorQuadrature(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_element: 0, index_quad: 1}
    _free_positions = [index_element]

#==============================================================================
class LocalTensorQuadrature(ArrayNode):
    # TODO add set_positions
    """
    """
    _rank = 1
    _positions = {index_quad: 0}

#==============================================================================
class TensorQuadrature(ScalarNode):
    """
    """
    pass

#==============================================================================
class MatrixQuadrature(MatrixNode):
    """
    """
    _rank = rank_dim

    def __new__(cls, target):
        # TODO check target
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]


#==============================================================================
class GlobalTensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 4
    _positions = {index_quad: 3, index_deriv: 2, index_dof: 1, index_element: 0}
    _free_positions = [index_element]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class LocalTensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 3
    _positions = {index_quad: 2, index_deriv: 1, index_dof: 0}
    _free_positions = [index_dof]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class TensorQuadratureBasis(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_quad: 1, index_deriv: 0}
    _free_positions = [index_quad]

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class CoefficientBasis(ScalarNode):
    """
    """
    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class TensorBasis(CoefficientBasis):
    pass

#==============================================================================
class GlobalTensorQuadratureTestBasis(GlobalTensorQuadratureBasis):
    _positions = {index_quad: 3, index_deriv: 2, index_dof_test: 1, index_element: 0}

#==============================================================================
class LocalTensorQuadratureTestBasis(LocalTensorQuadratureBasis):
    _positions = {index_quad: 2, index_deriv: 1, index_dof_test: 0}
    _free_positions = [index_dof_test]

#==============================================================================
class TensorQuadratureTestBasis(TensorQuadratureBasis):
    pass

#==============================================================================
class TensorTestBasis(TensorBasis):
    pass

#==============================================================================
class GlobalTensorQuadratureTrialBasis(GlobalTensorQuadratureBasis):
    _positions = {index_quad: 3, index_deriv: 2, index_dof_trial: 1, index_element: 0}

#==============================================================================
class LocalTensorQuadratureTrialBasis(LocalTensorQuadratureBasis):
    _positions = {index_quad: 2, index_deriv: 1, index_dof_trial: 0}
    _free_positions = [index_dof_trial]

#==============================================================================
class TensorQuadratureTrialBasis(TensorQuadratureBasis):
    pass

#==============================================================================
class TensorTrialBasis(TensorBasis):
    pass

#==============================================================================
class MatrixLocalBasis(MatrixNode):
    """
    used to describe local dof over an element
    """
    _rank = rank_dim

    def __new__(cls, target):
        # TODO check target
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class StencilMatrixLocalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil matrix
    """
    def __new__(cls, pads):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        return Basic.__new__(cls, pads, rank)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

#==============================================================================
class StencilVectorLocalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, pads):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        return Basic.__new__(cls, pads, rank)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

#==============================================================================
class StencilMatrixGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil matrix
    """
    def __new__(cls, pads):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = 2*len(pads)
        return Basic.__new__(cls, pads, rank)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

#==============================================================================
class StencilVectorGlobalBasis(MatrixNode):
    """
    used to describe local dof over an element as a stencil vector
    """
    def __new__(cls, pads):
        if not isinstance(pads, (list, tuple, Tuple)):
            raise TypeError('Expecting an iterable')

        pads = Tuple(*pads)
        rank = len(pads)
        return Basic.__new__(cls, pads, rank)

    @property
    def pads(self):
        return self._args[0]

    @property
    def rank(self):
        return self._args[1]

#==============================================================================
class GlobalSpan(ArrayNode):
    """
    """
    _rank = 1
    _positions = {index_element: 0}

    def __new__(cls, target):
        if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
            raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class Span(ScalarNode):
    """
    """
    def __new__(cls, target=None):
        if not( target is None ):
            if not isinstance(target, (ScalarTestFunction, VectorTestFunction)):
                raise TypeError('Expecting a scalar/vector test function')

        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class Evaluation(BaseNode):
    """
    """
    pass

#==============================================================================
class FieldEvaluation(Evaluation):
    """
    """
    pass

#==============================================================================
class MappingEvaluation(Evaluation):
    """
    """
    pass

#==============================================================================
class ComputeNode(Basic):
    """
    """
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class ComputePhysical(ComputeNode):
    """
    """
    pass

#==============================================================================
class ComputePhysicalBasis(ComputePhysical):
    """
    """
    pass

#==============================================================================
class ComputeLogical(ComputeNode):
    """
    """
    pass

#==============================================================================
class ComputeLogicalBasis(ComputeLogical):
    """
    """
    pass

#==============================================================================
class Accumulate(Basic):
    """
    """
    def __new__(cls, op, expr, lhs=None):
        # TODO add verification on op = '-', '+', '*', '/'
        return Basic.__new__(cls, op, expr, lhs)

    @property
    def op(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

    @property
    def lhs(self):
        return self._args[2]

#==============================================================================
class ElementOf(Basic):
    """
    """
    def __new__(cls, target):
        return Basic.__new__(cls, target)

    @property
    def target(self):
        return self._args[0]

#==============================================================================
class ExprNode(Basic):
    """
    """
    pass

#==============================================================================
class AtomicNode(ExprNode, AtomicExpr):
    """
    """

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class ValueNode(ExprNode):
    """
    """
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class PhysicalValueNode(ValueNode):
    """
    """
    pass

#==============================================================================
class LogicalValueNode(ValueNode):
    """
    """
    pass

#==============================================================================
class PhysicalBasisValue(PhysicalValueNode):
    """
    """
    pass

#==============================================================================
class LogicalBasisValue(LogicalValueNode):
    """
    """
    pass

#==============================================================================
class PhysicalGeometryValue(PhysicalValueNode):
    """
    """
    pass

#==============================================================================
class LogicalGeometryValue(LogicalValueNode):
    """
    """
    pass

#==============================================================================
class BasisAtom(AtomicNode):
    """
    """
    def __new__(cls, expr):
        ls  = list(expr.atoms(ScalarTestFunction))
        ls += list(expr.atoms(VectorTestFunction))
        if not(len(ls) == 1):
            print(expr, type(expr))
            print(ls)
            raise ValueError('Expecting an expression with one test function')

        u = ls[0]

        obj = Basic.__new__(cls, expr)
        obj._atom = u
        return obj

    @property
    def expr(self):
        return self._args[0]

    @property
    def atom(self):
        return self._atom

#==============================================================================
class GeometryAtom(AtomicNode):
    """
    """
    def __new__(cls, expr):
        ls = list(expr.atoms(Mapping))
        if not(len(ls) == 1):
            print(expr, type(expr))
            print(ls)
            raise ValueError('Expecting an expression with one mapping')

        # TODO
        u = ls[0]

        obj = Basic.__new__(cls, expr)
        obj._atom = u
        return obj

    @property
    def expr(self):
        return self._args[0]

    @property
    def atom(self):
        return self._atom


#==============================================================================
class Loop(BaseNode):
    """
    class to describe a loop of an iterator over a generator.
    """

    def __new__(cls, iterator, generator, stmts=None):
        # TODO stmts should not be optional
        # ...
        if isinstance(iterator, IteratorBase):
            iterator = [iterator]

        if not( isinstance(iterator, (list, tuple, Tuple)) ):
            raise TypeError('Expecting an iterable')

        if not all([isinstance(i, IteratorBase) for i in iterator]):
            raise TypeError('Expecting a list of Iterator')

        iterator = Tuple(*iterator)
        # ...

        # ...
        if isinstance(generator, GeneratorBase):
            generator = [generator]

        if not( isinstance(generator, (list, tuple, Tuple)) ):
            raise TypeError('Expecting an iterable')

        if not all([isinstance(i, GeneratorBase) for i in generator]):
            raise TypeError('Expecting a list of Generator')

        generator = Tuple(*generator)
        # ...

        # ...
        if stmts is None:
            stmts = []

        elif not isinstance(stmts, (tuple, list, Tuple)):
            stmts = [stmts]

        stmts = Tuple(*stmts)
        # ...

        return Basic.__new__(cls, iterator, generator, stmts)

    @property
    def iterator(self):
        return self._args[0]

    @property
    def generator(self):
        return self._args[1]

    @property
    def stmts(self):
        return self._args[2]

#==============================================================================
class TensorIteration(BaseNode):
    """
    """

    def __new__(cls, iterator, generator):
        # ...
        if not( isinstance(iterator, TensorIterator) ):
            raise TypeError('Expecting an TensorIterator')

        if not( isinstance(generator, TensorGenerator) ):
            raise TypeError('Expecting a TensorGenerator')
        # ...

        return Basic.__new__(cls, iterator, generator)

    @property
    def iterator(self):
        return self._args[0]

    @property
    def generator(self):
        return self._args[1]

#==============================================================================
class ProductIteration(BaseNode):
    """
    """

    def __new__(cls, iterator, generator):
        # ...
        if not( isinstance(iterator, ProductIterator) ):
            raise TypeError('Expecting an ProductIterator')

        if not( isinstance(generator, ProductGenerator) ):
            raise TypeError('Expecting a ProductGenerator')
        # ...

        return Basic.__new__(cls, iterator, generator)

    @property
    def iterator(self):
        return self._args[0]

    @property
    def generator(self):
        return self._args[1]

#==============================================================================
class SplitArray(BaseNode):
    """
    """
    def __new__(cls, target, positions, lengths):
        if not isinstance(positions, (list, tuple, Tuple)):
            positions = [positions]
        positions = Tuple(*positions)

        if not isinstance(lengths, (list, tuple, Tuple)):
            lengths = [lengths]
        lengths = Tuple(*lengths)

        return Basic.__new__(cls, target, positions, lengths)

    @property
    def target(self):
        return self._args[0]

    @property
    def positions(self):
        return self._args[1]

    @property
    def lengths(self):
        return self._args[2]

#==============================================================================
def construct_logical_expressions(u, nderiv):
    dim = u.space.ldim

    ops = [dx1, dx2, dx3][:dim]
    r = range(nderiv+1)
    ranges = [r]*dim
    indices = product(*ranges)

    indices = list(indices)
    indices = [ijk for ijk in indices if sum(ijk) <= nderiv]

    args = []
    for ijk in indices:
        atom = u
        for n,op in zip(ijk, ops):
            for i in range(1, n+1):
                atom = op(atom)
        args.append(atom)

    return [ComputeLogicalBasis(i) for i in args]

#==============================================================================
def construct_geometry_iter_gener(M, nderiv):
    dim = M.rdim

    ops = [dx1, dx2, dx3][:dim]
    r = range(nderiv+1)
    ranges = [r]*dim
    indices = product(*ranges)

    indices = list(indices)
    indices = [ijk for ijk in indices if sum(ijk) <= nderiv]

    args = []
    for d in range(dim):
        for ijk in indices:
            atom = M[d]
            for n,op in zip(ijk, ops):
                for i in range(1, n+1):
                    atom = op(atom)
            args.append(atom)

    iterators   = [ProductIterator(GeometryAtom(i))
                   for i in args]
    generators  = [ProductGenerator(MatrixQuadrature(i), index_quad)
                   for i in args]

    return iterators, generators
