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
class Grid(Generator):
    """
    """
    pass

#==============================================================================
class Element(Iterator):
    """
    """
    pass

#==============================================================================
class GlobalQuadrature(Generator):
    """
    """
    pass

#==============================================================================
class LocalQuadrature(Iterator, Generator):
    """
    """
    pass

#==============================================================================
class Quadrature(Iterator):
    """
    """
    pass

#==============================================================================
class GlobalBasis(Generator):
    """
    """
    pass

#==============================================================================
class LocalBasis(Iterator, Generator):
    """
    """
    pass

#==============================================================================
class Basis(Iterator):
    """
    """
    pass

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
