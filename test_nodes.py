# -*- coding: UTF-8 -*-

from sympy import Symbol
from sympy import Mul

from pyccel.ast import Assign
from pyccel.ast import AugAssign
# TODO remove
from pyccel.codegen.printing.pycode import pycode

from sympde.calculus import grad, dot
from sympde.topology import (dx, dy, dz)
from sympde.topology import (dx1, dx2, dx3)
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square
from sympde.topology import Mapping

from nodes import Grid
from nodes import Element
from nodes import TensorIterator
from nodes import TensorGenerator
from nodes import ProductIterator
from nodes import ProductGenerator
from nodes import Loop
from nodes import GlobalTensorQuadrature
from nodes import LocalTensorQuadrature
from nodes import TensorQuadrature
from nodes import MatrixQuadrature
from nodes import GlobalTensorQuadratureBasis
from nodes import LocalTensorQuadratureBasis
from nodes import TensorQuadratureBasis
from nodes import TensorBasis
from nodes import GlobalSpan
from nodes import Span
from nodes import BasisAtom
from nodes import PhysicalBasisValue
from nodes import LogicalBasisValue
from nodes import index_element
from nodes import index_quad
from nodes import index_dof
#from nodes import ComputePhysical
#from nodes import ComputeLogical
from nodes import ComputePhysicalBasis
from nodes import ComputeLogicalBasis
#from nodes import Accumulate # TODO fix
from nodes import construct_logical_expressions
from nodes import GeometryAtom
from nodes import PhysicalGeometryValue
from nodes import LogicalGeometryValue
from nodes import construct_geometry_iter_gener
from nodes import AtomicNode
from nodes import MatrixLocalBasis
from nodes import CoefficientBasis

from parser import parse

# ... abstract model
domain = Square()
M      = Mapping('M', domain.dim)

V      = ScalarFunctionSpace('V', domain)
u,v    = elements_of(V, names='u,v')
expr   = dot(grad(v), grad(u))
# ...

# ...
grid    = Grid()
element = Element()
g_quad  = GlobalTensorQuadrature()
l_quad  = LocalTensorQuadrature()
quad    = TensorQuadrature()
g_basis = GlobalTensorQuadratureBasis(u)
l_basis = LocalTensorQuadratureBasis(u)
a_basis = TensorQuadratureBasis(u)
basis   = TensorBasis(u)
g_span  = GlobalSpan(u)
span    = Span(u)
coeff   = CoefficientBasis(u)
l_coeff = MatrixLocalBasis(u)
# ...

#==============================================================================
def test_basis_atom_2d_1():
    expr = dx(u)
    lhs  = BasisAtom(expr)
    rhs  = PhysicalBasisValue(expr)

    settings = {'dim': domain.dim, 'nderiv': 1}
    _parse = lambda expr: parse(expr, settings=settings)

    u_x  = Symbol('u_x')
    u_x1 = Symbol('u_x1')

    assert(lhs.atom == u)
    assert(_parse(lhs) == u_x)
    assert(_parse(rhs) == u_x1)

#==============================================================================
def test_basis_atom_2d_2():
    expr = dy(dx(u))
    lhs  = BasisAtom(expr)
    rhs  = PhysicalBasisValue(expr)

    settings = {'dim': domain.dim, 'nderiv': 1}
    _parse = lambda expr: parse(expr, settings=settings)

    u_xy   = Symbol('u_xy')
    u_x1x2 = Symbol('u_x1x2')

    assert(lhs.atom == u)
    assert(_parse(lhs) == u_xy)
    assert(_parse(rhs) == u_x1x2)

#==============================================================================
def test_geometry_atom_2d_1():
    expr = M[0]
    lhs  = GeometryAtom(expr)
    rhs  = PhysicalGeometryValue(expr)

    settings = {'dim': domain.dim, 'nderiv': 1, 'mapping': M}
    _parse = lambda expr: parse(expr, settings=settings)

    x = Symbol('x')

    assert(_parse(lhs) == x)
    # TODO add assert on parse rhs

#==============================================================================
def test_loop_local_quad_2d_1():
    stmts = []
    iterator  = TensorIterator(quad)
    generator = TensorGenerator(l_quad, index_quad)
    loop      = Loop(iterator, generator, stmts)

    stmt = parse(loop, settings={'dim': domain.dim})
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_local_dof_quad_2d_1():
    # ...
    stmts = []
    iterator  = (TensorIterator(quad),
                 TensorIterator(basis))
    generator = (TensorGenerator(l_quad, index_quad),
                 TensorGenerator(a_basis, index_quad))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # ...
    stmts = [loop]
    iterator  = TensorIterator(a_basis)
    generator = TensorGenerator(l_basis, index_dof)
    loop      = Loop(iterator, generator, stmts)
    # ...

    # TODO bug when nderiv=0
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 1})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_local_dof_quad_2d_2():
    # ...
    args   = [dx(u), dx(dy(u)), dy(dy(u)), dx(u) + dy(u)]
    stmts  = [ComputePhysicalBasis(i) for i in args]
#    stmts += [Accumulate('+', dy(u)*dx(u))]
    # ...

    # ...
    iterator  = (TensorIterator(quad),
                 TensorIterator(basis))
    generator = (TensorGenerator(l_quad, index_quad),
                 TensorGenerator(a_basis, index_quad))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # ...
    stmts = [loop]
    iterator  = TensorIterator(a_basis)
    generator = TensorGenerator(l_basis, index_dof)
    loop      = Loop(iterator, generator, stmts)
    # ...

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_local_dof_quad_2d_3():
    # ...
    stmts  = [dx1(u)]
    stmts  = [ComputeLogicalBasis(i) for i in stmts]
    # ...

    # ...
    iterator  = (TensorIterator(quad),
                 TensorIterator(basis))
    generator = (TensorGenerator(l_quad, index_quad),
                 TensorGenerator(a_basis, index_quad))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # ...
    stmts = [loop]
    iterator  = TensorIterator(a_basis)
    generator = TensorGenerator(l_basis, index_dof)
    loop      = Loop(iterator, generator, stmts)
    # ...

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_local_dof_quad_2d_4():
    # ...
    stmts = []

    expressions  = [dx1(u), dx2(u)]
    stmts += [ComputeLogicalBasis(i) for i in expressions]

    expressions  = [dx(u)]
    stmts += [ComputePhysicalBasis(i) for i in expressions]
    # ...

    # ...
    iterator  = (TensorIterator(quad),
                 TensorIterator(basis))
    generator = (TensorGenerator(l_quad, index_quad),
                 TensorGenerator(a_basis, index_quad))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # ...
    stmts = [loop]
    iterator  = TensorIterator(a_basis)
    generator = TensorGenerator(l_basis, index_dof)
    loop      = Loop(iterator, generator, stmts)
    # ...

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_loop_global_local_quad_2d_1():
    # ...
    stmts = []
    iterator  = TensorIterator(quad)
    generator = TensorGenerator(l_quad, index_quad)
    loop      = Loop(iterator, generator, stmts)
    # ...

    # ...
    stmts = [loop]
    iterator  = TensorIterator(l_quad)
    generator = TensorGenerator(g_quad, index_element)
    loop      = Loop(iterator, generator, stmts)
    # ...

    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_span_2d_1():
    # ...
    stmts = []
    iterator  = TensorIterator(span)
    generator = TensorGenerator(g_span, index_element)
    loop      = Loop(iterator, generator, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_span_2d_1():
    # ...
    stmts = []
    iterator  = (TensorIterator(l_quad),
                 TensorIterator(span))
    generator = (TensorGenerator(g_quad, index_element),
                 TensorGenerator(g_span, index_element))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_basis_span_2d_1():
    # ...
    stmts = []
    iterator  = (TensorIterator(l_quad),
                 TensorIterator(l_basis),
                 TensorIterator(span))
    generator = (TensorGenerator(g_quad, index_element),
                 TensorGenerator(g_basis, index_element),
                 TensorGenerator(g_span, index_element))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_global_quad_basis_span_2d_2():
    # ...
    nderiv = 2
    stmts = construct_logical_expressions(u, nderiv)

    expressions = [dx(u), dx(dy(u)), dy(dy(u)), dx(u) + dy(u)]
    stmts  += [ComputePhysicalBasis(i) for i in expressions]
    # ...

    # ...
    iterator  = (TensorIterator(quad),
                 TensorIterator(basis))
    generator = (TensorGenerator(l_quad, index_quad),
                 TensorGenerator(a_basis, index_quad))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # ...
    stmts = [loop]
    iterator  = TensorIterator(a_basis)
    generator = TensorGenerator(l_basis, index_dof)
    loop      = Loop(iterator, generator, stmts)
    # ...

    # ...
    stmts = [loop]
    iterator  = (TensorIterator(l_quad),
                 TensorIterator(l_basis),
                 TensorIterator(span))
    generator = (TensorGenerator(g_quad, index_element),
                 TensorGenerator(g_basis, index_element),
                 TensorGenerator(g_span, index_element))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': nderiv})
    print(pycode(stmt))
    print()


#==============================================================================
def test_loop_local_quad_geometry_2d_1():
    # ...
    stmts = []
    geo_iterators, geo_generators = construct_geometry_iter_gener(M, nderiv=1)

    iterator  = [TensorIterator(quad)] + geo_iterators
    generator = [TensorGenerator(l_quad, index_quad)] + geo_generators
    loop      = Loop(iterator, generator, stmts)
    # ...

    settings = {'dim': domain.dim, 'nderiv': 1, 'mapping': M}
    _parse = lambda expr: parse(expr, settings=settings)

    stmt = _parse(loop)
    print()
    print(pycode(stmt))

    print()

#==============================================================================
def test_eval_field_2d_1():
    # ...
    args = [dx1(u), dx2(u)]

    # TODO improve
    stmts = [AugAssign(ProductGenerator(MatrixQuadrature(i), index_quad),
                                        '+', Mul(coeff,AtomicNode(i)))
             for i in args]
    # ...

    # ...
    nderiv = 1
    body = construct_logical_expressions(u, nderiv)
    # ...

    # ...
    stmts = body + stmts
    iterator  = (TensorIterator(quad),
                 TensorIterator(basis))
    generator = (TensorGenerator(l_quad, index_quad),
                 TensorGenerator(a_basis, index_quad))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # ...
    stmts = [loop]
    iterator  = (TensorIterator(a_basis),
                 ProductIterator(coeff))
    generator = (TensorGenerator(l_basis, index_dof),
                 ProductGenerator(l_coeff, index_dof))
    loop      = Loop(iterator, generator, stmts)
    # ...

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': nderiv})
    print(pycode(stmt))
    print()


#==============================================================================
# CLEAN UP SYMPY NAMESPACE
#==============================================================================

def teardown_module():
    from sympy import cache
    cache.clear_cache()

def teardown_function():
    from sympy import cache
    cache.clear_cache()


#==============================================================================

# tests without assert
test_loop_local_quad_2d_1()
test_loop_local_dof_quad_2d_1()
test_loop_local_dof_quad_2d_2()
test_loop_local_dof_quad_2d_3()
test_loop_local_dof_quad_2d_4()
test_loop_global_local_quad_2d_1()
test_global_span_2d_1()
test_global_quad_span_2d_1()
test_global_quad_basis_span_2d_1()
test_global_quad_basis_span_2d_2()
test_loop_local_quad_geometry_2d_1()
test_eval_field_2d_1()

# tests with assert
test_basis_atom_2d_1()
test_basis_atom_2d_2()
test_geometry_atom_2d_1()
