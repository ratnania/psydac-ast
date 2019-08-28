# -*- coding: UTF-8 -*-

from sympy import Symbol

from pyccel.ast import Assign

from sympde.calculus import grad, dot
from sympde.topology import (dx, dy, dz)
from sympde.topology import ScalarFunctionSpace
from sympde.topology import element_of, elements_of
from sympde.topology import Square

from nodes import Grid
from nodes import Element
from nodes import Loop
from nodes import GlobalQuadrature
from nodes import LocalQuadrature
from nodes import Quadrature
from nodes import GlobalBasis
from nodes import LocalBasis
from nodes import Basis
from nodes import BasisAtom
from nodes import BasisValue

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
g_basis = GlobalBasis()
l_basis = LocalBasis()
basis   = Basis()
# ...

#==============================================================================
def test_nodes_2d_1():
    print('============== test_nodes_2d_1 ===============')

    # ...
    loop = Loop(quad, l_quad)
    print(loop)
    # ...

    # ...
    loop = Loop(l_quad, g_quad, [loop])
    print(loop)
    # ...

    # ... TODO improve
    loop = Loop(element, grid, [loop])
    print(loop)
    # ...

#==============================================================================
def test_nodes_2d_2():
    print('============== test_nodes_2d_2 ===============')

    # ...
    loop = Loop(quad, l_quad)
    print(loop)
    # ...

    # ...
    loop = Loop(l_basis, g_basis, [loop])
    print(loop)
    # ...

    # ... TODO improve
    loop = Loop(element, grid, [loop])
    print(loop)
    # ...

#==============================================================================
def test_nodes_2d_3():
    print('============== test_nodes_2d_3 ===============')

    expr = dx(u)
    lhs  = BasisAtom(expr)
    rhs  = BasisValue(expr)

    assert(lhs.atom == u)
    assert(parse(lhs) == Symbol('u_x'))

    stmt = Assign(lhs, rhs)
    stmt = parse(stmt, dim=domain.dim, tests=[v], trials=[u])
    print(stmt)

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
test_nodes_2d_3()
