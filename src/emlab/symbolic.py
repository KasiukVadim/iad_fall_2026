"""Three symbolic representations and dependency graphs."""
import json
import time
import sympy as s
import networkx as nx
from sympy.vector import CoordSys3D, curl as vcurl, divergence as vdiv

x,y,z,t = s.symbols('x y z t', real=True)
XYZ = (x,y,z)

def grad(f):
    return s.Matrix([s.diff(f,q) for q in XYZ])

def div(v):
    return sum(s.diff(v[i],q) for i,q in enumerate(XYZ))

def curl(v):
    return s.Matrix([s.diff(v[2],y)-s.diff(v[1],z),s.diff(v[0],z)-s.diff(v[2],x),s.diff(v[1],x)-s.diff(v[0],y)])

def models():
    eps,mu,sigma = s.symbols('epsilon mu sigma', positive=True)
    E = s.Matrix([s.Function('E'+a)(x,y,z,t) for a in 'xyz'])
    H = s.Matrix([s.Function('H'+a)(x,y,z,t) for a in 'xyz'])
    J = s.Matrix([s.Function('J'+a)(x,y,z,t) for a in 'xyz'])
    return {'ampere':eps*E.diff(t)+sigma*E+J-curl(H),
            'faraday':mu*H.diff(t)+curl(E),
            'gauss_B':div(mu*H), 'gauss_D':div(eps*E)-s.Function('rho')(x,y,z,t)}

def expression_graph(expr):
    """Occurrence AST: edges parent -> operand, repeated nodes remain distinct."""
    g=nx.DiGraph()
    def visit(e):
        i=len(g);g.add_node(i,label=str(e) if not e.args else e.func.__name__)
        for arg in e.args:
            j=visit(arg);g.add_edge(i,j)
        return i
    visit(expr)
    return g

def export_models(out):
    eq=models()
    dictionary={k:([s.srepr(v) for v in e] if isinstance(e,s.MatrixBase) else s.srepr(e)) for k,e in eq.items()}
    (out/'equations.json').write_text(json.dumps(dictionary,indent=2))
    g=expression_graph(eq['ampere'][0]); nx.write_graphml(g,out/'expression.graphml')
    nodes=list(g);import numpy as np
    np.savetxt(out/'adjacency.csv',nx.to_numpy_array(g,nodelist=nodes,dtype=int),delimiter=',',fmt='%d')
    (out/'graph_nodes.json').write_text(json.dumps(dict(g.nodes(data='label')),indent=2))
    return g

def compare_backends():
    """Compare exact polynomial results; timing is a local microbenchmark."""
    f=s.Matrix([y*z,x*x*z,x*y*y]); expected=curl(f)
    C=CoordSys3D('C');fv=C.y*C.z*C.i+C.x**2*C.z*C.j+C.x*C.y**2*C.k
    a=time.perf_counter();v=vcurl(fv);elapsed=time.perf_counter()-a
    result={'sympy_matrix_curl':str(expected),'sympy_vector_curl':str(v),
            'sympy_vector_div':str(vdiv(fv)),'vector_seconds':elapsed,
            'matrix_vector_curl_equal': all(s.simplify(expected[i]-v.dot(b).subs({C.x:x,C.y:y,C.z:z}))==0 for i,b in enumerate([C.i,C.j,C.k]))}
    try:
        import symengine as se
        a=time.perf_counter(); xx=se.Symbol('x');p=(xx+1)**12
        result.update(symengine_derivative_equal=s.expand(s.sympify(str(se.diff(p,xx)),locals={'x':x})-s.diff((x+1)**12,x))==0,
                      symengine_seconds=time.perf_counter()-a)
    except ImportError:
        result['symengine']='optional dependency unavailable'
    return result
