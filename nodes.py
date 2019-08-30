from collections import OrderedDict

from sympy import Basic
from sympy.core.singleton import Singleton
from sympy.core.compatibility import with_metaclass
from sympy.core.containers import Tuple

from sympde.topology import ScalarTestFunction, VectorTestFunction

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

class IndexPoint(IndexNode):
    pass

class IndexDof(IndexNode):
    pass

class IndexDerivative(IndexNode):
    pass

index_element = IndexElement()
index_point   = IndexPoint()
index_dof     = IndexDof()
index_deriv   = IndexDerivative()

#==============================================================================
class LengthNode(with_metaclass(Singleton, Basic)):
    """Base class representing one length of an iterator"""
    pass

class LengthElement(LengthNode):
    pass

class LengthPoint(LengthNode):
    pass

class LengthDof(LengthNode):
    pass

length_element = LengthElement()
length_point   = LengthPoint()
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
class Iterator(BaseNode):
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
class Generator(BaseNode):
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

#==============================================================================
class LocalQuadrature(ArrayNode):
    # TODO add set_positions
    """
    """
    _rank = 1
    _positions = {index_point: 0}

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
    # TODO add index derivative
    _positions = {index_point: 2, index_deriv: 1, index_dof: 0}
    _free_positions = [index_point, index_dof]

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
class Compute(Basic):
    """
    """
    def __new__(cls, expr, op=None):
        # TODO add verification on op = '-', '+', '*', '/', None
        return Basic.__new__(cls, expr, op)

    @property
    def expr(self):
        return self._args[0]

    @property
    def op(self):
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
    pass

#==============================================================================
class BasisAtom(AtomicNode):
    """
    """
    def __new__(cls, expr):
        # ...
        ls  = list(expr.atoms(ScalarTestFunction))
        ls += list(expr.atoms(VectorTestFunction))
        if not(len(ls) == 1):
            print(expr)
            print(ls)
            raise ValueError('Expecting an expression with one test function')

        u = ls[0]
        # ...

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
class BasisValue(ValueNode):
    """
    """
    def __new__(cls, expr):
        return Basic.__new__(cls, expr)

    @property
    def expr(self):
        return self._args[0]

#==============================================================================
class Loop(BaseNode):
    """
    class to describe a loop of an iterator over a generator.
    """

    def __new__(cls, iterator, generator, stmts=None):
        # ...
        if not( isinstance(iterator, Iterator) ):
            raise TypeError('Expecting an Iterator')

        if not( isinstance(generator, Generator) ):
            raise TypeError('Expecting a Generator')
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
class EnumerateLoop(BaseNode):
    """
    class to describe an enumerated loop.
    """

    def __new__(cls, indices, lengths, iterator, iterable, stmts):
        # TODO sympy conform for indices, iterator, interable
        # ...
        if not isinstance(stmts, (tuple, list, Tuple)):
            raise TypeError('stmts must be a tuple, list or Tuple')

        stmts = Tuple(*stmts)
        # ...

        return Basic.__new__(cls, indices, lengths, iterator, iterable, stmts)

    @property
    def indices(self):
        return self._args[0]

    @property
    def lengths(self):
        return self._args[1]

    @property
    def iterator(self):
        return self._args[2]

    @property
    def iterable(self):
        return self._args[3]

    @property
    def stmts(self):
        return self._args[4]

#==============================================================================
class LoopGlobalQuadrature(Loop):
    """
    """
    def __new__(cls, stmts):
        g_quad  = GlobalQuadrature()
        l_quad  = LocalQuadrature()

        iterator  = Iterator(l_quad)
        generator = Generator(g_quad, index_element)

        return Loop.__new__(cls, iterator, generator, stmts)

#==============================================================================
class LoopLocalQuadrature(Loop):
    """
    """
    def __new__(cls, stmts):
        l_quad  = LocalQuadrature()
        quad    = Quadrature()

        iterator  = Iterator(quad)
        generator = Generator(l_quad, index_point)

        return Loop.__new__(cls, iterator, generator, stmts)

#==============================================================================
class LoopGlobalBasis(Loop):
    """
    """
    def __new__(cls, target, stmts):
        g_quad  = GlobalBasis(target)
        l_quad  = LocalBasis(target)

        iterator  = Iterator(l_quad)
        # TODO
        generator = Generator(g_quad, index_element)

        return Loop.__new__(cls, iterator, generator, stmts)

#==============================================================================
class LoopLocalBasis(Loop):
    """
    """
    def __new__(cls, target, stmts):
        l_quad  = LocalBasis(target)
        quad    = Basis(target)

        iterator  = Iterator(quad)
        generator = Generator(l_quad, index_dof)

        return Loop.__new__(cls, iterator, generator, stmts)

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
