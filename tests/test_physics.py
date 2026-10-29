import numpy as np
import sympy as s
import pytest
from pathlib import Path
from emlab.symbolic import x,y,z,grad,curl,div,expression_graph,models
from emlab.maxwell import simulate,plane_wave_error,curl as numerical_curl,divergence
from emlab.scattering import mie_qsca,rayleigh_qsca
from emlab.coupling import mutual_inductance,transient,HybridModel
from emlab.data import load_material,refractive_index

def test_symbolic_identities():
    f=x*x*y+s.sin(z);v=s.Matrix([x*y,y*z,z*x])
    assert curl(grad(f))==s.zeros(3,1)
    assert s.simplify(div(curl(v)))==0
    assert s.simplify(div(v.cross(grad(f)))-(grad(f).dot(curl(v))-v.dot(curl(grad(f)))))==0

def test_ast_roundtrip_and_adjacency():
    e=models()['ampere'][0];g=expression_graph(e)
    import networkx as nx
    A=nx.to_numpy_array(g,nodelist=list(g))
    assert nx.is_directed_acyclic_graph(g)
    assert A.sum()==len(g)-1
    assert s.sympify(s.srepr(e))==e

def test_discrete_div_curl():
    a=np.random.default_rng(8).normal(size=(3,10,10,10))
    assert np.max(abs(divergence(numerical_curl(a,.1),.1)))<1e-11

def test_maxwell_constraints_energy():
    r=simulate(n=12,steps=80)
    assert max(r.divD)<1e-10 and max(r.divH)<1e-10
    assert max(abs(r.energy/r.energy[0]-1))<1e-5

def test_second_order_wave():
    e1=plane_wave_error(12,96);e2=plane_wave_error(24,192)
    assert 3.5<e1/e2<4.5

def test_mie_rayleigh_limit_and_invisible_sphere():
    assert abs(mie_qsca(1.46,.01)/rayleigh_qsca(1.46,.01)-1)<1e-3
    assert mie_qsca(1.,1.)<1e-25

def test_coil_reciprocity_and_orientation():
    a=mutual_inductance(r1=.04,r2=.015,n=120)
    b=mutual_inductance(r1=.015,r2=.04,n=120)
    assert np.isclose(a,b,rtol=1e-10)
    assert abs(mutual_inductance(tilt=np.pi/2,n=120))<1e-18
    assert abs(a-mutual_inductance(n=240))/a<1e-6

def test_passivity_and_hybrid_baseline():
    t=np.linspace(0,1e-7,30)
    with pytest.raises(ValueError):transient(t,1e-6,4)
    h=HybridModel();a=h.predict(t);b=transient(t,mutual_inductance(),4.)
    assert np.allclose(a,b)

def test_material_ranges_and_values():
    root=Path(__file__).resolve().parents[1]/'data/raw'
    for name,expected in [('silica',1.46),('water',1.333)]:
        doc,_=load_material(name,root)
        assert abs(float(refractive_index(doc,.589))-expected)<.01
        with pytest.raises(ValueError):refractive_index(doc,.001)

def test_independent_symbolic_backends_agree():
    from emlab.symbolic import compare_backends
    r=compare_backends()
    assert r['matrix_vector_curl_equal']
    if 'symengine_derivative_equal' in r:assert r['symengine_derivative_equal']

def test_inverse_fit_recovers_known_parameters():
    from emlab.coupling import fit_transient
    t=np.linspace(0,8e-7,121);M=mutual_inductance()
    obs=transient(t,M,4.,rtol=1e-10)[3]
    fit=fit_transient(t,obs,1e-4)
    assert np.isclose(fit.x[0]*1e-9,M,rtol=.002)
    assert np.isclose(fit.x[1],4.,rtol=.002)
