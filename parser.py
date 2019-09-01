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
from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import SymbolicExpr
from sympde.topology import LogicalExpr
from sympde.topology import IdentityMapping
from sympde.topology.derivatives import get_index_logical_derivatives
from sympde.topology import element_of
from sympde.topology import ScalarTestFunction, VectorTestFunction
from sympde.expr.evaluation import _split_test_function

from nodes import AtomicNode
from nodes import BasisAtom
from nodes import PhysicalBasisValue
from nodes import LogicalBasisValue
from nodes import TensorQuadrature
from nodes import TensorBasis
from nodes import GlobalTensorQuadrature
from nodes import LocalTensorQuadrature
from nodes import LocalTensorQuadratureBasis
from nodes import TensorQuadratureBasis
from nodes import index_quad, length_quad
from nodes import index_dof, length_dof
from nodes import index_element, length_element
from nodes import index_deriv
from nodes import SplitArray
from nodes import Accumulate
from nodes import TensorIterationStatement


#==============================================================================
# TODO move it
import string
import random
def random_string( n ):
    chars    = string.ascii_lowercase + string.digits
    selector = random.SystemRandom()
    return ''.join( selector.choice( chars ) for _ in range( n ) )

#==============================================================================
_length_of_registery = {index_quad:    length_quad,
                        index_dof:     length_dof,
                        index_element: length_element, }

#==============================================================================
class Parser(object):
    """
    """
    def __init__(self, settings=None):
        # use a local copy (because of pop)
        if not settings is None:
            settings = settings.copy()

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

        # ...
        # TODO dim must be available !
        mapping = None
        if not( settings is None ):
            mapping = settings.pop('mapping', None)

        if mapping is None:
            mapping = IdentityMapping('Map', dim)

        self._mapping = mapping
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

    @property
    def mapping(self):
        return self._mapping

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
        # TODO do we need this?
#        lhs = self._visit(expr.lhs)
#        rhs = self._visit(expr.rhs)

        lhs = expr.lhs
        rhs = expr.rhs

        # ... extract slices from rhs
        slices = []
        if isinstance(rhs, IndexedElement):
            indices = rhs.indices[0]
            slices = [i for i in indices if isinstance(i, Slice)]
        # ...

        # ... update lhs with slices
        if len(slices) > 0:
            # TODO add assert on type lhs
            if isinstance(lhs, (IndexedBase, IndexedVariable)):
                lhs = lhs[slices]

            elif isinstance(lhs, Symbol):
                lhs = IndexedBase(lhs.name)[slices]

            else:
                raise NotImplementedError('{}'.format(type(lhs)))
        # ...

        return Assign(lhs, rhs)

    # ....................................................
    def _visit_AugAssign(self, expr):
        raise NotImplementedError('TODO')

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
    def _visit_GlobalTensorQuadrature(self, expr):
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
    def _visit_LocalTensorQuadrature(self, expr):
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
    def _visit_TensorQuadrature(self, expr):
        dim = self.dim

        names   = 'x1:%s'%(dim+1)
        points  = variables(names, dtype='real', cls=Variable)

        names   = 'w1:%s'%(dim+1)
        weights = variables(names, dtype='real', cls=Variable)

        target = list(zip(points, weights))
        return target

    # ....................................................
    def _visit_GlobalTensorQuadratureBasis(self, expr):
        # TODO add label
        # TODO add ln
        dim = self.dim
        rank = expr.rank
        ln = 1
        if ln > 1:
            names = 'global_basis_1:%s(1:%s)'%(dim+1,ln+1)
        else:
            names = 'global_basis_1:%s'%(dim+1)

        target = variables(names, dtype='real', rank=rank, cls=IndexedVariable)
        if not isinstance(target[0], (tuple, list, Tuple)):
            target = [target]
        target = list(zip(*target))
        return target

    # ....................................................
    def _visit_LocalTensorQuadratureBasis(self, expr):
        # TODO add label
        # TODO add ln
        dim = self.dim
        rank = expr.rank
        ln = 1
        if ln > 1:
            names = 'local_basis_1:%s(1:%s)'%(dim+1,ln+1)
        else:
            names = 'local_basis_1:%s'%(dim+1)

        target = variables(names, dtype='real', rank=rank, cls=IndexedVariable)
        if not isinstance(target[0], (tuple, list, Tuple)):
            target = [target]
        target = list(zip(*target))
        return target

    # ....................................................
    def _visit_TensorQuadratureBasis(self, expr):
        # TODO add label
        # TODO add ln
        dim = self.dim
        rank = expr.rank
        ln = 1
        if ln > 1:
            names = 'array_basis_1:%s(1:%s)'%(dim+1,ln+1)
        else:
            names = 'array_basis_1:%s'%(dim+1)

        target = variables(names, dtype='real', rank=rank, cls=IndexedVariable)
        if not isinstance(target[0], (tuple, list, Tuple)):
            target = [target]
        target = list(zip(*target))
        return target

    # ....................................................
    def _visit_TensorBasis(self, expr):
        # TODO label
        dim = self.dim
        nderiv = self.nderiv
        target = expr.target

        ops = [dx1, dx2, dx3][:dim]
        atoms =  _split_test_function(target)

        args = []
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
    def _visit_GlobalSpan(self, expr):
        dim = self.dim
        rank = expr.rank

        names  = 'global_span1:%s'%(dim+1)
        target = variables(names, dtype='int', rank=rank, cls=IndexedVariable)

        if not isinstance(target[0], (tuple, list, Tuple)):
            target = [target]
        target = list(zip(*target))
        return target

    # ....................................................
    def _visit_Span(self, expr):
        dim = self.dim

        names  = 'span1:%s'%(dim+1)
        target = variables(names, dtype='int', cls=Variable)

        if not isinstance(target[0], (tuple, list, Tuple)):
            target = [target]
        target = list(zip(*target))
        return target

    # ....................................................
    def _visit_Accumulate(self, expr):
        op   = expr.op
        expr = expr.expr

        # TODO improve lhs
        lhs = random_string( 6 )
        lhs = Symbol('tmp_{}'.format(lhs))

        rhs = self._visit(BasisValue(expr))
        return self._visit(AugAssign(lhs, op, rhs))

    # ....................................................
    def _visit_ComputeLogical(self, expr):
        expr = expr.expr
        if not isinstance(expr, (Add, Mul)):
            lhs = self._visit(AtomicNode(expr))
        else:
            lhs = random_string( 6 )
            lhs = Symbol('tmp_{}'.format(lhs))

        rhs = self._visit(LogicalValueNode(expr))
        return self._visit(Assign(lhs, rhs))

    # ....................................................
    def _visit_ComputePhysical(self, expr):
        expr = expr.expr
        if not isinstance(expr, (Add, Mul)):
            lhs = self._visit(AtomicNode(expr))
        else:
            lhs = random_string( 6 )
            lhs = Symbol('tmp_{}'.format(lhs))

        rhs = self._visit(PhysicalValueNode(expr))
        return self._visit(Assign(lhs, rhs))

    # ....................................................
    def _visit_ComputeLogicalBasis(self, expr):
        expr = expr.expr
        if not isinstance(expr, (Add, Mul)):
            lhs = self._visit(BasisAtom(expr))
        else:
            lhs = random_string( 6 )
            lhs = Symbol('tmp_{}'.format(lhs))

        rhs = self._visit(LogicalBasisValue(expr))
        return self._visit(Assign(lhs, rhs))

    # ....................................................
    def _visit_ComputePhysicalBasis(self, expr):
        expr = expr.expr
        if not isinstance(expr, (Add, Mul)):
            lhs = self._visit(BasisAtom(expr))
        else:
            lhs = random_string( 6 )
            lhs = Symbol('tmp_{}'.format(lhs))

        rhs = self._visit(PhysicalBasisValue(expr))
        return self._visit(Assign(lhs, rhs))

    # ....................................................
    def _visit_AtomicNode(self, expr):
        symbol = SymbolicExpr(expr.expr)
        return symbol

    # ....................................................
    def _visit_LogicalBasisValue(self, expr):
        # ...
        dim = self.dim
        coords = ['x1', 'x2', 'x3'][:dim]

        expr   = expr.expr
        atom   = BasisAtom(expr).atom
        atoms  = _split_test_function(atom)

        ops = [dx1, dx2, dx3][:dim]
        d_atoms = dict(zip(coords, atoms))
        d_ops   = dict(zip(coords, ops))
        d_indices = get_index_logical_derivatives(expr)
        args = []
        for k,u in d_atoms.items():
            d = d_ops[k]
            n = d_indices[k]
            for i in range(n):
                u = d(u)
            args.append(u)
        # ...

        expr = Mul(*args)
        return SymbolicExpr(expr)

    # ....................................................
    def _visit_PhysicalValueNode(self, expr):
        mapping = self.mapping
        expr = LogicalExpr(mapping, expr.expr)

        return SymbolicExpr(expr)

    # ....................................................
    def _visit_PhysicalGeometryValue(self, expr):
        mapping = self.mapping
        expr = LogicalExpr(mapping, expr.expr)

        return SymbolicExpr(expr)

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
    def _visit_IndexQuadrature(self, expr):
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
    def _visit_LengthQuadrature(self, expr):
        dim = self.dim
        return symbols('k1:%d'%(dim+1))

    # ....................................................
    def _visit_LengthDof(self, expr):
        # TODO must be p+1
        dim = self.dim
        return symbols('p1:%d'%(dim+1))

    # ....................................................
    def _visit_TensorIterator(self, expr):
        dim  = self.dim
        target = self._visit(expr.target)

        if expr.dummies is None:
            return target

        else:
            raise NotImplementedError('TODO')

    # ....................................................
    def _visit_TensorGenerator(self, expr):
        dim    = self.dim
        target = self._visit(expr.target)

        if expr.dummies is None:
            return target

#        if isinstance(expr.target, GlobalTensorQuadrature):
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
    def _visit_TensorIterationStatement(self, expr):
        iterator  = self._visit(expr.iterator)
        generator = self._visit(expr.generator)

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
                assert(isinstance(expr.generator.target, (LocalTensorQuadratureBasis, TensorQuadratureBasis)))

                positions = [expr.generator.target.positions[i] for i in [index_deriv]]
                args = []
                for xs in g_xs:
                    # TODO improve
                    a = SplitArray(xs, positions, [self.nderiv+1])
                    args += self._visit(a)
                g_xs = args

            for l_x,g_x in zip(l_xs, g_xs):
                # TODO improve
                if isinstance(l_x, IndexedVariable):
                    lhs = l_x

                elif isinstance(expr.generator.target, (LocalTensorQuadratureBasis, TensorQuadratureBasis)):
                    lhs = self._visit(BasisAtom(l_x))
                else:
                    lhs = l_x

                ls += [self._visit(Assign(lhs, g_x))]
            inits.append(ls)
        # ...

        return  indices, lengths, inits

    # ....................................................
    def _visit_Loop(self, expr):
#        print('**** Enter Loop ')
        # create iteration statements
        iterations = [TensorIterationStatement(i,j)
                      for i,j in zip(expr.iterator, expr.generator)]

        iterations = [self._visit(i) for i in iterations]
        indices, lengths, inits = zip(*iterations)
        # indices and lengths are suppose to be repeated here
        # we only take the first occurence
        indices = indices[0]
        lengths = lengths[0]

        inits_0 = inits[0]
        dim = self.dim
        for init in inits[1:]:
            for i in range(dim):
                inits_0[i] += init[i]

        inits = inits_0

#        print('> indices = ', indices)
#        print('> lengths = ', lengths)
#        print('> inits   = ', inits  )
#        import sys; sys.exit(0)

        # check if there is an accumulation
        # TODO init accumulation vars
#        accumulations = expr.stmts.atoms(Accumulate)

        # ...
        stmts = self._visit(expr.stmts)
        body  = list(stmts)
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

        # remove the list and return the For Node only
        body = body[0]

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


