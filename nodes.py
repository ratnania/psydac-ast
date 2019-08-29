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
class BaseNode(Basic):
    """
    """
    pass

#==============================================================================
class Iterator(BaseNode):
    """
    """
    pass

#==============================================================================
class Generator(BaseNode):
    """
    """
    pass

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
            raise TypeError('stmts must be a tuple, list or Tuple')

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
class Grid(Generator):
    """
    """
    pass

#==============================================================================
class Element(Iterator):
    """
    """
    def __new__(cls, grid):
        if not( isinstance(grid, Grid) ):
            raise TypeError('Expecting a Grid')

        return Basic.__new__(cls, grid)

    @property
    def grid(self):
        return self._args[0]

#==============================================================================
class GlobalQuadrature(Generator):
    """
    """
    def __new__(cls, grid):
        if not( isinstance(grid, Grid) ):
            raise TypeError('Expecting a Grid')

        return Basic.__new__(cls, grid)

    @property
    def grid(self):
        return self._args[0]

#==============================================================================
class LocalQuadrature(Iterator, Generator):
    """
    """
    _rank = 1
    def __new__(cls, element):
        if not( isinstance(element, Element) ):
            raise TypeError('Expecting a Element')

        return Basic.__new__(cls, element)

    @property
    def parent(self):
        return self._args[0]

    @property
    def rank(self):
        return self._rank

#==============================================================================
class Quadrature(Iterator):
    """
    """
    def __new__(cls, quad):
        if not( isinstance(quad, LocalQuadrature) ):
            raise TypeError('Expecting a LocalQuadrature')

        return Basic.__new__(cls, quad)

    @property
    def parent(self):
        return self._args[0]

#==============================================================================
class GlobalBasis(Generator):
    """
    """
    def __new__(cls, grid):
        if not( isinstance(grid, Grid) ):
            raise TypeError('Expecting a Grid')

        return Basic.__new__(cls, grid)

    @property
    def grid(self):
        return self._args[0]

#==============================================================================
class LocalBasis(Iterator, Generator):
    """
    """
    _rank = 3
    def __new__(cls, element):
        if not( isinstance(element, Element) ):
            raise TypeError('Expecting a Element')

        return Basic.__new__(cls, element)

    @property
    def parent(self):
        return self._args[0]

    @property
    def rank(self):
        return self._rank

#==============================================================================
class Basis(Iterator):
    """
    """
    def __new__(cls, basis):
        if not( isinstance(basis, LocalQuadrature) ):
            raise TypeError('Expecting a LocalQuadrature')

        return Basic.__new__(cls, basis)

    @property
    def parent(self):
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
class CartesianProjection(BaseNode):
    """
    """
    def __new__(cls, expr, index):
        return Basic.__new__(cls, expr, index)

    @property
    def expr(self):
        return self._args[0]

    @property
    def index(self):
        return self._args[1]
