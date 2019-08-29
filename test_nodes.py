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
from nodes import BasisAtom
from nodes import BasisValue
from nodes import index_element
from nodes import index_point
from nodes import index_dof
from nodes import LoopLocalQuadrature
from nodes import LoopGlobalQuadrature
from nodes import LoopLocalBasis
from nodes import LoopGlobalBasis

from parser import parse
from parser import index_of

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
# ...

#==============================================================================
def test_nodes_2d_1():
    # ...
    loop = LoopLocalQuadrature([])
    print(loop)
    # ...

    # ...
    loop = LoopGlobalQuadrature(loop)
    print(loop)
    # ...

    print()

#==============================================================================
def test_nodes_2d_2():
    # ...
    loop = LoopLocalBasis(u, [])
    print(loop)
    # ...

    # ...
    loop = LoopGlobalBasis(u, loop)
    print(loop)
    # ...

    print()

#==============================================================================
def test_nodes_2d_3():
    expr = dx(u)
    lhs  = BasisAtom(expr)
    rhs  = BasisValue(expr)

    assert(lhs.atom == u)
    assert(parse(lhs) == Symbol('u_x'))

    stmt = Assign(lhs, rhs)
    indices = {'basis': index_of(basis, domain.dim),
               'quad':  index_of(quad, domain.dim)}
    settings = {'dim':     domain.dim,
                'tests':   [v],
                'trials':  [u],
                'indices': indices}
    stmt = parse(stmt, settings)
    print(stmt)

    print()

#==============================================================================
def test_nodes_2d_4():
    expr = dx(u)
    lhs  = BasisAtom(expr)
    rhs  = BasisValue(expr)

    assert(lhs.atom == u)
    assert(parse(lhs) == Symbol('u_x'))

    indices = {'basis': index_of(basis, domain.dim),
               'quad':  index_of(quad, domain.dim)}

    stmt = Assign(lhs, rhs)
    loop = LoopLocalQuadrature(stmt)
    settings = {'dim':     domain.dim,
                'tests':   [v],
                'trials':  [u],
                'indices': indices}
    print(loop)
    stmt = parse(loop, settings)
    print(pycode(stmt))

#    stmt = parse(stmt, dim=domain.dim, tests=[v], trials=[u], indices=indices)
#    print(stmt)
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

test_nodes_2d_1()
test_nodes_2d_2()
test_nodes_2d_3()
#test_nodes_2d_4()
