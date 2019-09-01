from collections import OrderedDict
from itertools import product

from sympy import Basic
from sympy.core.singleton import Singleton
from sympy.core.compatibility import with_metaclass
from sympy.core.containers import Tuple

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

class IndexDerivative(IndexNode):
    pass

index_element = IndexElement()
index_quad    = IndexQuadrature()
index_dof     = IndexDof()
index_deriv   = IndexDerivative()

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

length_element = LengthElement()
length_quad    = LengthQuadrature()
length_dof     = LengthDof()

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
class IteratorNode(BaseNode):
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
class TensorIterator(IteratorNode):
    pass

#==============================================================================
class GeneratorNode(BaseNode):
    """
    """
    def __new__(cls, target, dummies):
        if not isinstance(dummies, (list, tuple, Tuple)):
            dummies = [dummies]
        dummies = Tuple(*dummies)

        if not isinstance(target, ArrayNode):
            raise TypeError('expecting an ArrayNode')

        return Basic.__new__(cls, target, dummies)

    @property
    def target(self):
        return self._args[0]

    @property
    def dummies(self):
        return self._args[1]

#==============================================================================
class TensorGenerator(GeneratorNode):
    pass

#==============================================================================
class Grid(BaseNode):
    """
    """
    pass

#==============================================================================
class ArrayNode(BaseNode):
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
class ScalarNode(BaseNode):
    """
    """
    pass

#==============================================================================
class GlobalQuadrature(ArrayNode):
    """
    """
    _rank = 2
    _positions = {index_element: 0, index_quad: 1}
    _free_positions = [index_element]

#==============================================================================
class LocalQuadrature(ArrayNode):
    # TODO add set_positions
    """
    """
    _rank = 1
    _positions = {index_quad: 0}

#==============================================================================
class Quadrature(ScalarNode):
    """
    """
    pass

#==============================================================================
class GlobalBasis(ArrayNode):
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
class LocalBasis(ArrayNode):
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
class ArrayBasis(ArrayNode):
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
class Basis(ScalarNode):
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
    def __new__(cls, target):
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
    def __new__(cls, op, expr):
        # TODO add verification on op = '-', '+', '*', '/'
        return Basic.__new__(cls, op, expr)

    @property
    def op(self):
        return self._args[0]

    @property
    def expr(self):
        return self._args[1]

#==============================================================================
class ExprNode(Basic):
    """
    """
    pass

#==============================================================================
class AtomicNode(ExprNode):
    """
    """
    pass

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
        if isinstance(iterator, IteratorNode):
            iterator = [iterator]

        if not( isinstance(iterator, (list, tuple, Tuple)) ):
            raise TypeError('Expecting an iterable')

        if not all([isinstance(i, IteratorNode) for i in iterator]):
            raise TypeError('Expecting a list of Iterator')

        iterator = Tuple(*iterator)
        # ...

        # ...
        if isinstance(generator, GeneratorNode):
            generator = [generator]

        if not( isinstance(generator, (list, tuple, Tuple)) ):
            raise TypeError('Expecting an iterable')

        if not all([isinstance(i, GeneratorNode) for i in generator]):
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
class TensorIterationStatement(BaseNode):
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
def loop_global_quadrature(stmts):
    """
    """
    g_quad  = GlobalQuadrature()
    l_quad  = LocalQuadrature()

    iterator  = TensorIterator(l_quad)
    generator = TensorGenerator(g_quad, index_element)

    return Loop(iterator, generator, stmts)

#==============================================================================
def loop_local_quadrature(stmts):
    """
    """
    l_quad  = LocalQuadrature()
    quad    = Quadrature()

    iterator  = TensorIterator(quad)
    generator = TensorGenerator(l_quad, index_quad)

    return Loop(iterator, generator, stmts)

#==============================================================================
def loop_global_basis(target, stmts):
    """
    """
    g_basis  = GlobalBasis(target)
    l_basis  = LocalBasis(target)

    iterator  = TensorIterator(l_basis)
    # TODO
    generator = TensorGenerator(g_basis, index_element)

    return Loop(iterator, generator, stmts)

#==============================================================================
def loop_local_basis(target, stmts):
    """
    """
    l_basis  = LocalBasis(target)
    a_basis    = ArrayBasis(target)

    iterator  = TensorIterator(a_basis)
    generator = TensorGenerator(l_basis, index_dof)

    return Loop(iterator, generator, stmts)

#==============================================================================
def loop_array_basis(target, stmts):
    """
    """
    a_basis  = ArrayBasis(target)
    basis    = Basis(target)

    iterator  = TensorIterator(basis)
    generator = TensorGenerator(a_basis, index_quad)

    return Loop(iterator, generator, stmts)

#==============================================================================
def loop_global_span(target, stmts):
    """
    """
    g_span = GlobalSpan(target)
    span   = Span(target)

    iterator  = TensorIterator(span)
    generator = TensorGenerator(g_span, index_element)

    return Loop(iterator, generator, stmts)

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
