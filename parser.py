from collections import OrderedDict

from sympy import IndexedBase, Indexed
from sympy import Mul
from sympy import Add
from sympy import symbols, Symbol
from sympy.core.containers import Tuple

from pyccel.ast import Range, Product, For
from pyccel.ast import Assign
from pyccel.ast import AugAssign
from pyccel.ast import Variable, IndexedVariable, IndexedElement
from pyccel.ast import Slice

from sympde.topology import (dx, dy, dz)
from sympde.topology import SymbolicExpr
from sympde.topology.derivatives import get_index_derivatives
from sympde.topology import element_of
#from sympde.expr.evaluation import _split_test_function # TODO use it

from nodes import BasisAtom
from nodes import BasisValue
from nodes import Quadrature
from nodes import Basis
from nodes import GlobalQuadrature
from nodes import LocalQuadrature
from nodes import LocalBasis
from nodes import index_point, length_point
from nodes import index_dof, length_dof
from nodes import index_element, length_element
from nodes import index_deriv
from nodes import SplitArray
from nodes import Accumulate


#==============================================================================
# TODO move it
import string
import random
def random_string( n ):
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

#==============================================================================
# TODO must be moved to sympde
# TODO re-test gelato
from sympde.topology import ScalarTestFunction
from sympde.topology import ScalarFunctionSpace
from sympde.topology import Interval
def _split_test_function(expr):

    if isinstance(expr, ScalarTestFunction):

        dim = expr.space.ldim
        coords = ['x', 'y', 'z'][:dim]
        coords = [Symbol(i) for i in coords]
        name = expr.name

        ls = []
        for i in range(0, dim):
            Di = Interval(coordinate=coords[i])
            Vi = ScalarFunctionSpace('tmp_V_{}'.format(i), domain=Di)

            ai = ScalarTestFunction(Vi, '{name}{i}'.format(name=name, i=i+1))
            ls += [ai]

        return ls

    # TODO
#    elif isinstance(expr, IndexedTestTrial):
#
#        i = expr.indices
#        assert(len(i) == 1)
#        i = i[0]
#
#        V = expr.base.space
#        Vi = ScalarFunctionSpace('tmpV_{}'.format(i), V.domain)
#        vi = ScalarTestFunction(Vi, '{test}{i}'.format(test=expr.base.name, i=i))
#
#        return _split_test_function(vi)

    else:
        msg = 'Expecting ScalarTestFunction or IndexedTestTrial, given {}'.format(type(expr))
        raise TypeError(msg)

#==============================================================================
_length_of_registery = {index_point:   length_point,
                        index_dof:     length_dof,
                        index_element: length_element, }

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

        # ...
        nderiv = None
        if not( settings is None ):
            nderiv = settings.pop('nderiv', None)
            if nderiv is None:
                raise ValueError('nderiv not provided')

        self._nderiv = nderiv
        # ...

        self._settings = settings

        # TODO improve
        self.free_indices = OrderedDict()
        self.free_lengths = OrderedDict()

    @property
    def settings(self):
        return self._settings

    @property
    def dim(self):
        return self._dim

    @property
    def nderiv(self):
        return self._nderiv

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

    # ....................................................
    def _visit_Tuple(self, expr):
        args = [self._visit(i) for i in expr]
        return Tuple(*args)

    # ....................................................
    def _visit_Grid(self, expr):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_Element(self, expr):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_GlobalQuadrature(self, expr):
        dim  = self.dim
        rank = expr.rank

        names = 'global_x1:%s'%(dim+1)
        points   = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        names = 'global_w1:%s'%(dim+1)
        weights  = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        # gather by axis
        target = list(zip(points, weights))

        return target

    # ....................................................
    def _visit_LocalQuadrature(self, expr):
        dim  = self.dim
        rank = expr.rank

        names = 'local_x1:%s'%(dim+1)
        points   = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        names = 'local_w1:%s'%(dim+1)
        weights  = variables(names, dtype='real', rank=rank, cls=IndexedVariable)

        # gather by axis
        target = list(zip(points, weights))

        return target

    # ....................................................
    def _visit_Quadrature(self, expr):
        dim = self.dim

        names   = 'x1:%s'%(dim+1)
        points  = variables(names, dtype='real', cls=Variable)

        names   = 'w1:%s'%(dim+1)
        weights = variables(names, dtype='real', cls=Variable)

        target = list(zip(points, weights))
        return target

    # ....................................................
    def _visit_GlobalBasis(self, expr):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_LocalBasis(self, expr):
        # TODO add label
        # TODO add ln
        dim = self.dim
        rank = expr.rank
        ln = 1
        if ln > 1:
            names = 'basis_1:%s(1:%s)'%(dim+1,ln+1)
        else:
            names = 'basis_1:%s'%(dim+1)

        target = variables(names, dtype='real', rank=rank, cls=IndexedVariable)
        if not isinstance(target[0], (tuple, list, Tuple)):
            target = [target]
        target = list(zip(*target))
        return target

    # ....................................................
    def _visit_Basis(self, expr):
        # TODO label
        dim = self.dim
        nderiv = self.nderiv
        target = expr.target
        ops = [dx, dy, dz][:dim]
        args = []
        atoms =  _split_test_function(target)
        for i,atom in enumerate(atoms):
            d = ops[i]
            ls = [atom]
            a = atom
            for n in range(1, nderiv+1):
                a = d(a)
                ls.append(a)

            args.append(ls)

        return args

    # ....................................................
    def _visit_Accumulate(self, expr):
        op   = expr.op
        expr = expr.expr

        # TODO improve lhs
        lhs = random_string( 6 )
        lhs = Symbol('tmp_{}'.format(lhs))

        rhs = self._visit(BasisValue(expr))
        return AugAssign(lhs, op, rhs)

    # ....................................................
    def _visit_Compute(self, expr):
        expr = expr.expr
        if not isinstance(expr, (Add, Mul)):
            lhs = self._visit(BasisAtom(expr))
        else:
            lhs = random_string( 6 )
            lhs = Symbol('tmp_{}'.format(lhs))

        rhs = self._visit(BasisValue(expr))
        return Assign(lhs, rhs)

    # ....................................................
    def _visit_FieldEvaluation(self, expr):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_MappingEvaluation(self, expr):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_BasisAtom(self, expr):
        symbol = SymbolicExpr(expr.expr)
        return symbol

    # ....................................................
    def _visit_BasisValue(self, expr):
        # ...
        expr   = expr.expr
        atom   = BasisAtom(expr).atom
        atoms  = _split_test_function(atom)
        # ...

        new = Mul(*atoms)
        expr = expr.subs({atom: new})
        return SymbolicExpr(expr)
    # ....................................................

    # ....................................................
    def _visit_Pattern(self, expr):
        # this is for multi-indices for the moment
        dim = self.dim
        args = []
        for a in expr:
            if a is None:
                args.append([Slice(None, None)]*dim)

            elif isinstance(a, int):
                args.append([a]*dim)

            else:
                v = self._visit(a)
                args.append(v)

        args = list(zip(*args))

        if len(args) == 1:
            args = args[0]

        return args

    # ....................................................
    def _visit_IndexElement(self, expr):
        dim = self.dim
        return symbols('i_element_1:%d'%(dim+1))

    # ....................................................
    def _visit_IndexPoint(self, expr):
        dim = self.dim
        return symbols('i_quad_1:%d'%(dim+1))

    # ....................................................
    def _visit_IndexDof(self, expr):
        dim = self.dim
        return symbols('i_basis_1:%d'%(dim+1))

    # ....................................................
    def _visit_IndexDerivative(self, expr):
        raise NotImplementedError('TODO')

    # ....................................................
    def _visit_LengthElement(self, expr):
        dim = self.dim
        return symbols('n_element_1:%d'%(dim+1))

    # ....................................................
    def _visit_LengthPoint(self, expr):
        dim = self.dim
        return symbols('k1:%d'%(dim+1))

    # ....................................................
    def _visit_LengthDof(self, expr):
        # TODO must be p+1
        dim = self.dim
        return symbols('p1:%d'%(dim+1))

    # ....................................................
    def _visit_Iterator(self, expr):
        dim  = self.dim
        target = self._visit(expr.target)

        if expr.dummies is None:
            return target

        else:
            raise NotImplementedError('TODO')

    # ....................................................
    def _visit_Generator(self, expr):
        dim    = self.dim
        target = self._visit(expr.target)

        if expr.dummies is None:
            return target

#        if isinstance(expr.target, GlobalQuadrature):
#            print('> target = ', target)
#            print('> dummies = ', expr.dummies)
#            for i in expr.dummies:
#                print(i, type(i))
#                a = self._visit(i)
#                print(i, '     ' , a)
#            import sys; sys.exit(0)

        # treat dummies and put them in the namespace
        dummies = self._visit(expr.dummies)
        dummies = list(zip(*dummies)) # TODO improve
        self.free_indices[expr.dummies] = dummies

        # add dummies as args of pattern()
        pattern = expr.target.pattern()
        pattern = self._visit(pattern)

        args = []
        for p, xs in zip(pattern, target):
            ls = []
            for x in xs:
                ls.append(x[p])
            args.append(ls)

        return args

    # ....................................................
    def _visit_Loop(self, expr):
#        print('**** Enter Loop ')
        iterator  = self._visit(expr.iterator)
        generator = self._visit(expr.generator)

        # check if there is an accumulation
        # TODO init accumulation vars
#        accumulations = expr.stmts.atoms(Accumulate)

        stmts     = self._visit(expr.stmts)

        dummies = expr.generator.dummies
        lengths = [_length_of_registery[i] for i in dummies]
        lengths = [self._visit(i) for i in lengths]
        lengths = list(zip(*lengths)) # TODO
        indices = self.free_indices[dummies]

        # ...
        inits = []
        for l_xs, g_xs in zip(iterator, generator):
            ls = []
            # there is a special case here,
            # when local var is a list while the global var is
            # an array of rank 1. In this case we want to enumerate all the
            # components of the global var.
            # this is the case when dealing with derivatives in each direction
            # TODO maybe we should add a flag here or a kwarg that says we
            # should enumerate the array
            if len(l_xs) > len(g_xs):
                assert(isinstance(expr.generator.target, LocalBasis))

                positions = [expr.generator.target.positions[i] for i in [index_deriv]]
                args = []
                for xs in g_xs:
                    # TODO improve
                    a = SplitArray(xs, positions, [self.nderiv+1])
                    args += self._visit(a)
                g_xs = args

            for l_x,g_x in zip(l_xs, g_xs):
                if isinstance(expr.generator.target, LocalBasis):
                    lhs = self._visit(BasisAtom(l_x))
                else:
                    lhs = l_x

                ls += [Assign(lhs, g_x)]
            inits.append(ls)
        # ...

        # ...
        body = list(stmts)
        for index, length, init in zip(indices, lengths, inits):
            if len(length) == 1:
                l = length[0]
                i = index[0]
                ranges = [Range(l)]

            else:
                ranges = [Range(l) for l in length]
                i = index

            body = init + body
            body = [For(i, Product(*ranges), body)]
        # ...

#        print('**** End   Loop ')
        return body

    # ....................................................
    def _visit_SplitArray(self, expr):
        target  = expr.target
        positions = expr.positions
        lengths = expr.lengths
        base = target.base

        args = []
        for p,n in zip(positions, lengths):
            indices = target.indices[0] # sympy is return a tuple of tuples
            indices = [i for i in indices] # make a copy
            for i in range(n):
                indices[p] = i
                x = base[indices]
                args.append(x)

        return args

    # ....................................................
    def _visit_IndexedElement(self, expr):
        return expr

    # ....................................................
    # TODO to be removed. usefull for testing
    def _visit_Pass(self, expr):
        return expr


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


