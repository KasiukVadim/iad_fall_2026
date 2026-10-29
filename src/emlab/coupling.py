"""3D coil geometry -> Neumann mutual inductance -> coupled RLC ODE.

SI units throughout. Quasi-static air coils at 13.56 MHz; this is a
reduced analogue of journal UHF designs, not their antenna reproduction.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
MU0=1.25663706212e-6

def loop(radius,center=(0,0,0),tilt=0,n=240):
    a=(np.arange(n)+.5)*2*np.pi/n
    u=np.array([1.,0,0]);v=np.array([0,np.cos(tilt),np.sin(tilt)])
    r=np.asarray(center)+radius*(np.cos(a)[:,None]*u+np.sin(a)[:,None]*v)
    dl=radius*2*np.pi/n*(-np.sin(a)[:,None]*u+np.cos(a)[:,None]*v)
    return r,dl

def mutual_inductance(r1=.04,r2=.015,d=.04,tilt=0,n=240):
    a,da=loop(r1,n=n);b,db=loop(r2,(0,0,d),tilt,n)
    dist=np.linalg.norm(a[:,None]-b[None,:],axis=-1)
    if dist.min()<1e-8:raise ValueError('Intersecting filaments')
    return MU0/(4*np.pi)*np.sum((da@db.T)/dist)

@dataclass
class Circuit:
    L1:float=1e-6
    L2:float=.5e-6
    R1:float=2.
    R2:float=4.
    frequency:float=13.56e6
    def capacitances(self):
        w=2*np.pi*self.frequency
        return 1/(w*w*self.L1),1/(w*w*self.L2)

def transient(times,M,R2,circuit=None,rtol=2e-8):
    c=circuit or Circuit();C1,C2=c.capacitances()
    if M*M>=c.L1*c.L2 or R2<=0:raise ValueError('Non-passive circuit')
    Linv=np.linalg.inv([[c.L1,M],[M,c.L2]])
    # State [q1/C1,q2/C2,i1,i2], capacitor voltage scaling avoids tiny charges.
    def rhs(t,y):
        v1,v2,i1,i2=y
        di=Linv@np.array([np.cos(2*np.pi*c.frequency*t)-c.R1*i1-v1,-R2*i2-v2])
        return [i1/C1,i2/C2,*di]
    sol=solve_ivp(rhs,(0,float(times[-1])),np.zeros(4),t_eval=times,rtol=rtol,atol=rtol*1e-2,method='DOP853')
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y

def fit_transient(times,observations,noise_std):
    def residual(p):return (transient(times,p[0]*1e-9,p[1])[3]-observations)/noise_std
    fit=least_squares(residual,[12.,6.],bounds=([.1,.5],[100.,15.]),x_scale='jac',ftol=1e-9,xtol=1e-9,gtol=1e-9)
    if not fit.success:raise RuntimeError(fit.message)
    return fit

class HybridModel:
    """Composable geometric-physical + learnable residual model.

    correction(features) is a user-injected scalar in henry. The passivity
    constraint is verified when the corrected M enters the circuit model.
    """
    def __init__(self,correction=None):self.correction=correction or (lambda features:0.)
    def predict(self,times,d=.04,tilt=0,R2=4.):
        M=mutual_inductance(d=d,tilt=tilt)+float(self.correction(np.array([d,tilt])))
        return transient(times,M,R2)
