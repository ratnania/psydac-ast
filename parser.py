from collections import OrderedDict

from sympy import IndexedBase
from sympy import Mul
from sympy import symbols, Symbol
from sympy.core.containers import Tuple

from pyccel.ast import Assign
from pyccel.ast.core import Variable, IndexedVariable

from sympde.topology import SymbolicExpr
from sympde.topology.derivatives import get_index_derivatives
from sympde.topology import element_of

from nodes import BasisAtom
from nodes import Quadrature
from nodes import Basis
from nodes import LocalQuadrature
from nodes import LocalBasis

#==============================================================================
def index_of(expr, dim):
    if isinstance(expr, Quadrature):
        return symbols('i_quad_1:%d'%(dim+1))

    elif isinstance(expr, Basis):
        return symbols('i_basis_1:%d'%(dim+1))

    else:
        raise NotImplementedError('TODO')

#==============================================================================
class Parser(object):
    """
    """
    def __init__(self, settings=None):
        # ...
        dim = None
        if not( settings is None ):
            dim = settings.pop('dim', None)
            if dim is None:
                raise ValueError('dim not provided')

        self._dim = dim
        # ...

        self._settings = settings

    @property
    def settings(self):
        return self._settings

    @property
    def dim(self):
        return self._dim

    def doit(self, expr, **settings):
        return self._visit(expr, **settings)

    def _visit(self, expr, **settings):
        classes = type(expr).__mro__
        for cls in classes:
            annotation_method = '_visit_' + cls.__name__
            if hasattr(self, annotation_method):
                return getattr(self, annotation_method)(expr, **settings)

        # Unknown object, we raise an error.
        raise NotImplementedError('{}'.format(type(expr)))

    # ....................................................
    def _visit_Assign(self, expr):
        lhs = self._visit(expr.lhs)
        rhs = self._visit(expr.rhs)

        return Assign(lhs, rhs)

    def _visit_Tuple(self, expr):
        args = [self._visit(i) for i in expr]
        return Tuple(*args)

    def _visit_Loop(self, expr):
        iterator  = self._visit(expr.iterator)
        generator = self._visit(expr.generator)
        stmts     = self._visit(expr.stmts)

        print('> iterator  = ', iterator)
        print('> generator = ', generator)
        print('> stmts     = ', stmts    )

        expr = expr # TODO
        return expr

    def _visit_Grid(self, expr):
        raise NotImplementedError('TODO')

    def _visit_Element(self, expr):
        raise NotImplementedError('TODO')

    def _visit_GlobalQuadrature(self, expr):
        raise NotImplementedError('TODO')

    def _visit_LocalQuadrature(self, expr):
        dim = self.dim

        names = 'points_1:%s'%(dim+1)
        points   = variables(names, dtype='real', rank=2, cls=IndexedVariable)

        names = 'weights_1:%s'%(dim+1)
        weights  = variables(names, dtype='real', rank=2, cls=IndexedVariable)

        return points, weights

    def _visit_Quadrature(self, expr):
        # TODO return a tuple? as if it was an enumerate
        indices = index_of(expr, self.dim)
        return indices

    def _visit_GlobalBasis(self, expr):
        raise NotImplementedError('TODO')

    def _visit_LocalBasis(self, expr):
        # TODO return a tuple? as if it was an enumerate
        dim = self.dim
        # TODO
        ln = 1

        if ln > 1:
            names = 'trial_basis_1:%s(1:%s)'%(dim+1,ln+1)
        else:
            names = 'trial_basis_1:%s'%(dim+1)

        basis = variables(names, dtype='real', rank=4, cls=IndexedVariable)

        return basis

    def _visit_Basis(self, expr):
        # TODO return a tuple? as if it was an enumerate
        indices = index_of(expr, self.dim)
        return indices

    def _visit_FieldEvaluation(self, expr):
        raise NotImplementedError('TODO')

    def _visit_MappingEvaluation(self, expr):
        raise NotImplementedError('TODO')

    def _visit_BasisAtom(self, expr):
        symbol = SymbolicExpr(expr.expr)
        return symbol

    def _visit_BasisValue(self, expr):
        # ...
        dim = self.dim
        settings = self.settings.copy()
        tests    = settings.pop('tests', [])
        trials   = settings.pop('trials', [])
        indices  = settings.pop('indices', None)

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
def parse(expr, settings=None):
    return Parser(settings).doit(expr)


#==============================================================================
# TODO should be imported from psydac
import re
import string
from sympy.utilities.iterables import cartes

_range = re.compile('([0-9]*:[0-9]+|[a-zA-Z]?:[a-zA-Z])')

def variables(names, dtype, **args):

    def contruct_variable(cls, name, dtype, rank, **args):
        if issubclass(cls, Variable):
            return Variable(dtype,  name, rank=rank, **args)
        elif issubclass(cls, IndexedVariable):
            return IndexedVariable(name, dtype=dtype, rank=rank, **args)
        else:
            raise TypeError('only Variables and IndexedVariables are supported')

    result = []
    cls = args.pop('cls', Variable)

    rank = args.pop('rank', 0)

    if isinstance(names, str):
        marker = 0
        literals = [r'\,', r'\:', r'\ ']
        for i in range(len(literals)):
            lit = literals.pop(0)
            if lit in names:
                while chr(marker) in names:
                    marker += 1
                lit_char = chr(marker)
                marker += 1
                names = names.replace(lit, lit_char)
                literals.append((lit_char, lit[1:]))
        def literal(s):
            if literals:
                for c, l in literals:
                    s = s.replace(c, l)
            return s

        names = names.strip()
        as_seq = names.endswith(',')
        if as_seq:
            names = names[:-1].rstrip()
        if not names:
            raise ValueError('no symbols given')

        # split on commas
        names = [n.strip() for n in names.split(',')]
        if not all(n for n in names):
            raise ValueError('missing symbol between commas')
        # split on spaces
        for i in range(len(names) - 1, -1, -1):
            names[i: i + 1] = names[i].split()

        seq = args.pop('seq', as_seq)

        for name in names:
            if not name:
                raise ValueError('missing variable')

            if ':' not in name:
                var = contruct_variable(cls, literal(name), dtype, rank, **args)
                result.append(var)
                continue

            split = _range.split(name)
            # remove 1 layer of bounding parentheses around ranges
            for i in range(len(split) - 1):
                if i and ':' in split[i] and split[i] != ':' and \
                        split[i - 1].endswith('(') and \
                        split[i + 1].startswith(')'):
                    split[i - 1] = split[i - 1][:-1]
                    split[i + 1] = split[i + 1][1:]
            for i, s in enumerate(split):
                if ':' in s:
                    if s[-1].endswith(':'):
                        raise ValueError('missing end range')
                    a, b = s.split(':')
                    if b[-1] in string.digits:
                        a = 0 if not a else int(a)
                        b = int(b)
                        split[i] = [str(c) for c in range(a, b)]
                    else:
                        a = a or 'a'
                        split[i] = [string.ascii_letters[c] for c in range(
                            string.ascii_letters.index(a),
                            string.ascii_letters.index(b) + 1)]  # inclusive
                    if not split[i]:
                        break
                else:
                    split[i] = [s]
            else:
                seq = True
                if len(split) == 1:
                    names = split[0]
                else:
                    names = [''.join(s) for s in cartes(*split)]
                if literals:
                    result.extend([contruct_variable(cls, literal(s), dtype, rank, **args) for s in names])
                else:
                    result.extend([contruct_variable(cls, s, dtype, rank, **args) for s in names])

        if not seq and len(result) <= 1:
            if not result:
                return ()
            return result[0]

        return tuple(result)
    elif isinstance(names,(tuple,list)):
        return tuple(variables(i, dtype, cls=cls,rank=rank,**args) for i in names)
    else:
        raise TypeError('Expecting a string')


