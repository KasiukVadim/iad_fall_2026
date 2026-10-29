"""Lossless nonmagnetic sphere in vacuum: exact Mie and Rayleigh limit."""
import numpy as np
from scipy.special import spherical_jn, spherical_yn

def mie_qsca(m,x):
    if x<=0 or m<=0: raise ValueError('m and x must be positive real numbers')
    n=np.arange(1,int(np.ceil(x+4*x**(1/3)+2))+1)
    def psi(z): return z*spherical_jn(n,z)
    def dp(z): return spherical_jn(n,z)+z*spherical_jn(n,z,derivative=True)
    px,pm=psi(x),psi(m*x);dx,dm=dp(x),dp(m*x)
    xi=x*(spherical_jn(n,x)+1j*spherical_yn(n,x))
    dxi=dx+1j*(spherical_yn(n,x)+x*spherical_yn(n,x,derivative=True))
    an=(m*pm*dx-px*dm)/(m*pm*dxi-xi*dm)
    bn=(pm*dx-m*px*dm)/(pm*dxi-m*xi*dm)
    return float(2/x**2*np.sum((2*n+1)*(abs(an)**2+abs(bn)**2)))

def rayleigh_qsca(m,x):
    return 8/3*np.asarray(x)**4*abs((m*m-1)/(m*m+2))**2
