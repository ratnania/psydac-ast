from collections import OrderedDict

from sympy import IndexedBase
from sympy import Mul
from sympy import symbols, Symbol

from pyccel.ast import Assign

from sympde.topology import SymbolicExpr
from sympde.topology.derivatives import get_index_derivatives
from sympde.topology import element_of

from nodes import BasisAtom
from nodes import LocalQuadrature
from nodes import LocalBasis


#==============================================================================
def index_of(expr, dim):
    if isinstance(expr, LocalQuadrature):
        return symbols('i_quad_1:%d'%(dim+1))

    elif isinstance(expr, LocalBasis):
        return symbols('i_basis_1:%d'%(dim+1))

    else:
        raise NotImplementedError('TODO')

#==============================================================================
class Parser(object):
    """
    """
    def __init__(self, settings=None):
        self._settings = settings

    @property
    def settings(self):
        return self._settings

    def doit(self, expr):
        return self._visit(expr)

    def _visit(self, expr, **kwargs):
        classes = type(expr).__mro__
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(expr, **kwargs)

        # Unknown object, we raise an error.
        raise NotImplementedError('{}'.format(type(expr)))

    # ....................................................
    def _visit_Assign(self, expr):
        lhs = self._visit(expr.lhs)
        rhs = self._visit(expr.rhs)

        return Assign(lhs, rhs)

    def _visit_Loop(self, expr):
        raise NotImplementedError('TODO')

    def _visit_Grid(self, expr):
        raise NotImplementedError('TODO')

    def _visit_Element(self, expr):
        raise NotImplementedError('TODO')

    def _visit_GlobalQuadrature(self, expr):
        raise NotImplementedError('TODO')

    def _visit_LocalQuadrature(self, expr):
        raise NotImplementedError('TODO')

    def _visit_Quadrature(self, expr):
        raise NotImplementedError('TODO')

    def _visit_GlobalBasis(self, expr):
        raise NotImplementedError('TODO')

    def _visit_LocalBasis(self, expr):
        raise NotImplementedError('TODO')

    def _visit_Basis(self, expr):
        raise NotImplementedError('TODO')

    def _visit_FieldEvaluation(self, expr):
        raise NotImplementedError('TODO')

    def _visit_MappingEvaluation(self, expr):
        raise NotImplementedError('TODO')

    def _visit_BasisAtom(self, expr):
        symbol = SymbolicExpr(expr.expr)
        return symbol

    def _visit_BasisValue(self, expr):
        # ...
        settings = self.settings.copy()
        dim      = settings.pop('dim', None)
        tests    = settings.pop('tests', [])
        trials   = settings.pop('trials', [])
        indices  = settings.pop('indices', None)

        if dim is None:
            raise ValueError('dim not provided')

        if ( indices is None ) or not isinstance(indices, (dict, OrderedDict)):
            raise ValueError('indices must be a dictionary')

        if not( 'quad' in indices.keys() ):
            raise ValueError('quad not provided for indices')

        if not( 'basis' in indices.keys() ):
            raise ValueError('basis not provided for indices')
        # ...

        # ...
        expr   = expr.expr
        atom   = BasisAtom(expr).atom
        orders = [*get_index_derivatives(expr).values()]
        # ...

        # ...
        if atom in tests:
            name = 'Ni'
        elif atom in trials:
            name = 'Nj'
        else:
            raise NotImplementedError('TODO')
        # ...

        # ...
        basis = [IndexedBase('{name}{index}'.format(name=name, index=i))
                 for i in range(dim)]
        # ...

        args = [b[i, d, q]
                for b, i, d, q in zip(basis, indices['basis'], orders, indices['quad'])]
        rhs = Mul(*args)

        return rhs
    # ....................................................

#==============================================================================
def parse(expr, **settings):
    return Parser(settings).doit(expr)
