# -*- coding: UTF-8 -*-

from sympy import Symbol

from pyccel.ast import Assign
# TODO remove
from pyccel.codegen.printing.pycode import pycode

from sympde.calculus import grad, dot
from sympde.topology import (dx, dy, dz)
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square

from nodes import Grid
from nodes import Element
from nodes import Iterator
from nodes import Generator
from nodes import Loop
from nodes import GlobalQuadrature
from nodes import LocalQuadrature
from nodes import Quadrature
from nodes import GlobalBasis
from nodes import LocalBasis
from nodes import Basis
from nodes import GlobalSpan
from nodes import Span
from nodes import BasisAtom
from nodes import BasisValue
from nodes import index_element
from nodes import index_point
from nodes import index_dof
from nodes import loop_local_quadrature
from nodes import loop_global_quadrature
from nodes import loop_local_basis
from nodes import loop_global_basis
from nodes import loop_global_span
from nodes import Compute
from nodes import Accumulate

from parser import parse

# ... abstract model
domain = Square()
V      = ScalarFunctionSpace('V', domain)
u,v    = elements_of(V, names='u,v')
expr   = dot(grad(v), grad(u))
# ...

# ...
grid    = Grid()
element = Element()
g_quad  = GlobalQuadrature()
l_quad  = LocalQuadrature()
quad    = Quadrature()
g_basis = GlobalBasis(u)
l_basis = LocalBasis(u)
basis   = Basis(u)
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
    rhs  = BasisValue(expr)

    assert(lhs.atom == u)
    assert(parse(lhs) == Symbol('u_x'))

    u1_x = Symbol('u1_x')
    u2   = Symbol('u2')
    assert(parse(rhs) == u1_x*u2)

    print()

#==============================================================================
def test_nodes_2d_3b():
    expr = dy(dx(u))
    lhs  = BasisAtom(expr)
    rhs  = BasisValue(expr)

    print(parse(lhs) )
    print(parse(rhs) )

    print()

#==============================================================================
def test_nodes_2d_4():
    loop = loop_local_quadrature([])
    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_5():
    loop = loop_local_quadrature([])
    loop = loop_local_basis(u, [loop])
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 3})
    print()
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_6():
#    body  = [dx(u), dx(dy(u)), dy(dy(u)), dx(u) + dy(u)]
#    body  = [Compute(i) for i in body]
#    body += [Accumulate('+', dy(u)*dx(u))]
    body  = [dx(u)]
    body  = [Compute(i) for i in body]
#    body += [Accumulate('+', dy(u)*dx(u))]
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
    iterator  = [Iterator(i) for i in iterator]

    generator  = (g_quad, g_span)
    generator  = [Generator(i, index_element) for i in generator]

    stmts = []
    loop = Loop(iterator, generator, stmts)

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_10():
    iterator  = (l_quad, l_basis, span)
    iterator  = [Iterator(i) for i in iterator]

    generator  = (g_quad, g_basis, g_span)
    generator  = [Generator(i, index_element) for i in generator]

    stmts = []
    loop = Loop(iterator, generator, stmts)

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
    print(pycode(stmt))
    print()

#==============================================================================
def test_nodes_2d_11():
    # ...
    body  = [dx(u), dx(dy(u)), dy(dy(u)), dx(u) + dy(u)]
    body  = [Compute(i) for i in body]
    body += [Accumulate('+', dy(u)*dx(u))]
    loop = loop_local_quadrature(body)
    loop = loop_local_basis(u, [loop])
    # ...

    iterator  = (l_quad, l_basis, span)
    iterator  = [Iterator(i) for i in iterator]

    generator  = (g_quad, g_basis, g_span)
    generator  = [Generator(i, index_element) for i in generator]

    stmts = [loop]
    loop = Loop(iterator, generator, stmts)

    # TODO do we need nderiv here?
    stmt = parse(loop, settings={'dim': domain.dim, 'nderiv': 2})
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

#test_nodes_2d_1()
#test_nodes_2d_2()
#test_nodes_2d_3a()
#test_nodes_2d_3b()
#test_nodes_2d_4()
#test_nodes_2d_5()
test_nodes_2d_6()
#test_nodes_2d_7()
#test_nodes_2d_8()
#test_nodes_2d_9()
#test_nodes_2d_10()
#test_nodes_2d_11()
