# -*- coding: UTF-8 -*-

from sympy import Symbol

from pyccel.ast import Assign
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
from nodes import loop_local_quadrature
from nodes import loop_global_quadrature
from nodes import loop_array_basis
from nodes import loop_local_basis
from nodes import loop_global_basis
from nodes import loop_global_span
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
# ...

#==============================================================================
def test_nodes_2d_1():
    # ...
    loop = loop_local_quadrature([])
    print(loop)
    # ...

    # ...
    loop = loop_global_quadrature(loop)
    print(loop)
    # ...

    print()

#==============================================================================
def test_nodes_2d_2():
    # ...
    loop = loop_local_basis(u, [])
    print(loop)
    # ...

    # ...
    loop = loop_global_basis(u, loop)
    print(loop)
    # ...

    print()

#==============================================================================
def test_nodes_2d_3a():
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

    print()

#==============================================================================
def test_nodes_2d_3b():
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

    print()

#==============================================================================
def test_nodes_2d_4():
    loop = loop_local_quadrature([])
    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
# TODO to remove: we should not allow such tree
#def test_nodes_2d_5a():
#    loop = loop_local_quadrature([])
#    loop = loop_array_basis(u, [loop])
#    # TODO bug when nderiv=0
#    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 1})
#    print()
#    print(pycode(stmt))
#    print()

#==============================================================================
def test_nodes_2d_5b():
    loop = loop_local_quadrature([])
    loop = loop_array_basis(u, [loop])
    loop = loop_local_basis(u, [loop])

    # TODO bug when nderiv=0
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 1})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_6a():
    body  = [dx(u), dx(dy(u)), dy(dy(u)), dx(u) + dy(u)]
    body  = [ComputePhysicalBasis(i) for i in body]
#    body += [Accumulate('+', dy(u)*dx(u))]
    loop = loop_local_quadrature(body)
    loop = loop_local_basis(u, [loop])
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_6b():
    body  = [dx1(u)]
    body  = [ComputeLogicalBasis(i) for i in body]
    loop = loop_local_quadrature(body)
    loop = loop_local_basis(u, [loop])
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_6c():
    body = []

    expressions  = [dx1(u), dx2(u)]
    body += [ComputeLogicalBasis(i) for i in expressions]

    expressions  = [dx(u)]
    body += [ComputePhysicalBasis(i) for i in expressions]

    loop = loop_local_quadrature(body)
    loop = loop_local_basis(u, [loop])
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_7():
    loop = loop_local_quadrature([])
    loop = loop_global_quadrature([loop])
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_8():
    loop = loop_global_span(u, [])
    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_9():
    iterator  = (l_quad, span)
    iterator  = [TensorIterator(i) for i in iterator]

    generator  = (g_quad, g_span)
    generator  = [TensorGenerator(i, index_element) for i in generator]

    stmts = []
    loop = Loop(iterator, generator, stmts)

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_10():
    iterator  = (l_quad, l_basis, span)
    iterator  = [TensorIterator(i) for i in iterator]

    generator  = (g_quad, g_basis, g_span)
    generator  = [TensorGenerator(i, index_element) for i in generator]

    stmts = []
    loop = Loop(iterator, generator, stmts)

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_11():
    # ...
    nderiv = 2
    body = construct_logical_expressions(u, nderiv)

    expressions = [dx(u), dx(dy(u)), dy(dy(u)), dx(u) + dy(u)]
    body  += [ComputePhysicalBasis(i) for i in expressions]

    loop = loop_local_quadrature(body)
    loop = loop_array_basis(u, [loop])
    loop = loop_local_basis(u, [loop])
    # ...

    iterator  = (l_quad, l_basis, span)
    iterator  = [TensorIterator(i) for i in iterator]

    generator  = (g_quad, g_basis, g_span)
    generator  = [TensorGenerator(i, index_element) for i in generator]

    stmts = [loop]
    loop = Loop(iterator, generator, stmts)

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': nderiv})
    print(pycode(stmt))
    print()



#==============================================================================
def test_nodes_2d_20a():
    expr = M[0]
    lhs  = GeometryAtom(expr)
    rhs  = PhysicalGeometryValue(expr)

    settings = {'dim': domain.dim, 'nderiv': 1, 'mapping': M}
    _parse = lambda expr: parse(expr, settings=settings)

    x = Symbol('x')

    assert(_parse(lhs) == x)
    # TODO add assert on parse rhs

    print()

#==============================================================================
def test_nodes_2d_20b():
    stmts = []

    geo_iterators, geo_generators = construct_geometry_iter_gener(M, nderiv=1)
    iterator  = [TensorIterator(quad)] + geo_iterators
    generator = [TensorGenerator(l_quad, index_quad)] + geo_generators

    loop = Loop(iterator, generator, stmts)

    settings = {'dim': domain.dim, 'nderiv': 1, 'mapping': M}
    _parse = lambda expr: parse(expr, settings=settings)

    stmt = _parse(loop)
    print()
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
#test_nodes_2d_20b()
#import sys; sys.exit(0)

# tests without assert
test_nodes_2d_1()
test_nodes_2d_2()
test_nodes_2d_4()
test_nodes_2d_5b()
test_nodes_2d_6a()
test_nodes_2d_6b()
test_nodes_2d_6c()
test_nodes_2d_7()
test_nodes_2d_8()
test_nodes_2d_9()
test_nodes_2d_10()
test_nodes_2d_11()
test_nodes_2d_20b()

# tests with assert
test_nodes_2d_3a()
test_nodes_2d_3b()
test_nodes_2d_20a()
