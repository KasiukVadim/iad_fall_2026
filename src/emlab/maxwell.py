"""3D periodic Maxwell IVP, normalized epsilon0=mu0=c=1.

Centered differences are skew-adjoint on a periodic grid. Integrate D,H
rather than E,H so that discrete div(D)=div(H)=0 are conserved by curl.
No PML/open-boundary or far-field RCS claim is made.
"""
from dataclasses import dataclass
import numpy as np

def derivative(a,axis,dx):return (np.roll(a,-1,axis=axis)-np.roll(a,1,axis=axis))/(2*dx)
def curl(a,dx):
    d=lambda i,j:derivative(a[i],j,dx)
    return np.array([d(2,1)-d(1,2),d(0,2)-d(2,0),d(1,0)-d(0,1)])
def divergence(a,dx):return sum(derivative(a[i],i,dx) for i in range(3))

@dataclass
class FieldResult:
    xyz: tuple
    times: np.ndarray
    E: np.ndarray
    H: np.ndarray
    energy: np.ndarray
    divD: np.ndarray
    divH: np.ndarray
    epsilon: np.ndarray

def simulate(n=24,contrast=2.1,steps=160,final_time=.65,sphere=True):
    if n<8 or steps<1 or contrast<1: raise ValueError('Invalid grid, steps or contrast')
    dx=1/n;dt=final_time/steps
    # RK4 stability on imaginary eigenvalues; conservative sufficient condition.
    if dt*np.sqrt(3)/dx>2.7:raise ValueError('RK4 stability bound exceeded')
    q=np.arange(n)*dx;X,Y,Z=np.meshgrid(q,q,q,indexing='ij')
    r=np.sqrt((X-.5)**2+(Y-.5)**2+(Z-.5)**2)
    eps=1+(contrast-1)*np.exp(-(r/.14)**8) if sphere else np.ones_like(X)
    # A periodic plane wave in D. Initial D,H are divergence-free.
    D=np.zeros((3,n,n,n));H=D.copy();D[1]=np.cos(2*np.pi*X);H[2]=np.cos(2*np.pi*X)
    def rhs(d,h):return curl(h,dx),-curl(d/eps,dx)
    def energy(d,h):return .5*np.mean(np.sum(d*d/eps+h*h,axis=0))
    frames=[];hs=[];times=[];en=[];dd=[];dh=[]
    for i in range(steps+1):
        if i%max(1,steps//20)==0 or i==steps:
            frames.append((D/eps).copy());hs.append(H.copy());times.append(i*dt)
            en.append(energy(D,H));dd.append(np.max(abs(divergence(D,dx))));dh.append(np.max(abs(divergence(H,dx))))
        if i==steps:break
        a,b=rhs(D,H);c,d=rhs(D+dt*a/2,H+dt*b/2);e,f=rhs(D+dt*c/2,H+dt*d/2);g,h=rhs(D+dt*e,H+dt*f)
        D+=dt*(a+2*c+2*e+g)/6;H+=dt*(b+2*d+2*f+h)/6
    return FieldResult((X,Y,Z),np.array(times),np.array(frames),np.array(hs),np.array(en),np.array(dd),np.array(dh),eps)

def plane_wave_error(n=24,steps=160,final_time=.65):
    r=simulate(n=n,steps=steps,final_time=final_time,sphere=False)
    exact=np.cos(2*np.pi*(r.xyz[0]-final_time))
    return float(np.sqrt(np.mean((r.E[-1,1]-exact)**2)))
