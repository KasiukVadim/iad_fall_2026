"""Run all experiments and regenerate every figure and numeric table."""
from pathlib import Path
from make_animation import render
import sys,json,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import least_squares
from emlab.symbolic import export_models,compare_backends
from emlab.data import load_material,refractive_index
from emlab.scattering import mie_qsca,rayleigh_qsca
from emlab.maxwell import simulate,plane_wave_error
from emlab.coupling import mutual_inductance,loop,transient,fit_transient
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.bbox':'tight'})
def save(name):plt.savefig(OUT/f'{name}.png');plt.close()

def main():
    rng=np.random.default_rng(20261029);metrics={};provenance=[]
    g=export_models(OUT);metrics['symbolic']=compare_backends()
    fig,ax=plt.subplots(figsize=(11,5));pos=nx.spring_layout(g,seed=3)
    nx.draw_networkx(g,pos,labels=nx.get_node_attributes(g,'label'),node_size=600,font_size=7,arrowsize=10,ax=ax,node_color='#c5e4f3');ax.set_title('Expression tree: x-component of Ampere residual');save('expression_graph')
    order=list(g);A=nx.to_numpy_array(g,nodelist=order)
    fig,ax=plt.subplots(figsize=(6,5));ax.imshow(A,cmap='Blues');ax.set(xlabel='Operand node ID',ylabel='Parent node ID',title='Directed adjacency matrix');save('adjacency')
    wavelengths=np.linspace(.45,1.0,150);fig,axs=plt.subplots(1,2,figsize=(11,4))
    for name in ['silica','water']:
        doc,p=load_material(name,ROOT/'data/raw');provenance.append(p);m=refractive_index(doc,wavelengths)
        x=2*np.pi*.025/wavelengths # radius 25 nm; wavelengths in micrometres
        qm=np.array([mie_qsca(mm,xx) for mm,xx in zip(m,x)]);qr=rayleigh_qsca(m,x)
        axs[0].plot(wavelengths,m,label=name);axs[1].plot(wavelengths,qm,label=f'{name}: Mie');axs[1].plot(wavelengths,qr,'--',label=f'{name}: Rayleigh')
        np.savetxt(OUT/f'{name}_scattering.csv',np.c_[wavelengths,m,qm,qr],delimiter=',',header='wavelength_um,n,Q_Mie,Q_Rayleigh',comments='')
        metrics[f'{name}_rayleigh_max_relative_error']=float(np.max(abs(qr-qm)/qm))
    axs[0].set(xlabel='Wavelength (um)',ylabel='Refractive index n',title='Two published material datasets');axs[1].set(xlabel='Wavelength (um)',ylabel='Q scattering',title='25 nm sphere in vacuum');[a.legend(fontsize=8) for a in axs];fig.tight_layout();save('materials_scattering')
    xs=np.geomspace(.01,2,100);m=1.46;qm=np.array([mie_qsca(m,x) for x in xs]);qr=rayleigh_qsca(m,xs)
    fig,ax=plt.subplots(figsize=(8,4));ax.loglog(xs,abs(qm-qr)/qm);ax.axhline(.05,ls='--',c='gray');ax.set(xlabel='Size parameter x=2 pi a / lambda',ylabel='Relative Rayleigh error',title='Applicability of a small-sphere approximation');save('rayleigh_error')
    print('Solving Maxwell 3D',flush=True);field=simulate();ref=simulate(sphere=False)
    np.savez_compressed(OUT/'fields_3d.npz',times=field.times,E=field.E,H=field.H,epsilon=field.epsilon,energy=field.energy,divD=field.divD,divH=field.divH)
    delta=field.E[-1]-ref.E[-1];fig,axs=plt.subplots(1,3,figsize=(12,3.6))
    for ax,arr,title in zip(axs,[field.epsilon,field.E[-1,1],delta[1]],['Permittivity, z=0.5','Total Ey at t=0.65','Ey minus homogeneous IVP']):
        im=ax.imshow(arr[:,:,12].T,origin='lower',extent=[0,1,0,1],cmap='RdBu_r');fig.colorbar(im,ax=ax,shrink=.75);ax.set(title=title,xlabel='x',ylabel='y')
    fig.tight_layout();save('maxwell_slices')
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(projection='3d');sl=(slice(None,None,4),)*3
    ax.quiver(*[v[sl] for v in field.xyz],*[v[sl] for v in delta],length=.1,normalize=False,color='#1678a3');ax.set(xlabel='x',ylabel='y',zlabel='z',title='3D vector perturbation E - E homogeneous');save('maxwell_3d')
    fig,axs=plt.subplots(1,2,figsize=(10,3.6));axs[0].plot(field.times,field.energy/field.energy[0]-1);axs[0].set(xlabel='Normalized time',ylabel='Relative energy drift')
    axs[1].plot(field.times,field.divD,label='max |div D|');axs[1].plot(field.times,field.divH,label='max |div H|');axs[1].legend();axs[1].set(xlabel='Normalized time',ylabel='Constraint residual');fig.tight_layout();save('conservation')
    sizes=[12,24,48];errs=[plane_wave_error(n=n,steps=8*n) for n in sizes]
    metrics.update(maxwell_energy_relative_drift=float(np.max(abs(field.energy/field.energy[0]-1))),maxwell_max_divD=float(max(field.divD)),maxwell_max_divH=float(max(field.divH)),plane_wave_grid_sizes=sizes,plane_wave_rmse=errs,plane_wave_orders=list(np.log2(np.array(errs[:-1])/errs[1:])))
    fig,ax=plt.subplots(figsize=(7,4));ax.loglog(sizes,errs,'o-',label='Numerical RMSE');ax.loglog(sizes,errs[0]*(np.array(sizes)/sizes[0])**-2.,'--',label='Second order');ax.set(xlabel='Grid points per axis',ylabel='Plane-wave RMSE');ax.legend();save('convergence')
    print('Fitting RLC transient',flush=True);M=mutual_inductance();times=np.linspace(0,2e-6,501);truth=transient(times,M,4.,rtol=1e-10)[3];noise=.01*np.max(abs(truth));obs=truth+rng.normal(0,noise,len(times))
    train=np.arange(len(times))%3!=0;fit=fit_transient(times[train],obs[train],noise)
    # transient starts at t=0 independently of first sampled observation.
    pred=transient(times,fit.x[0]*1e-9,fit.x[1])[3]
    metrics['coupling_fit']={'true_M_nH':M*1e9,'fitted_M_nH':fit.x[0],'true_R2_ohm':4.,'fitted_R2_ohm':fit.x[1],'heldout_rmse_A':float(np.sqrt(np.mean((pred[~train]-truth[~train])**2))),'noise_std_A':noise,'jacobian_rank':int(np.linalg.matrix_rank(fit.jac)),'nfev':fit.nfev}
    np.savetxt(OUT/'synthetic_transient.csv',np.c_[times,truth,obs,pred,train],delimiter=',',header='time_s,true_i2_A,noisy_i2_A,fitted_i2_A,is_train',comments='')
    fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].scatter(times[::4]*1e6,obs[::4]*1e3,s=8,alpha=.5,label='Synthetic noisy data');axs[0].plot(times*1e6,pred*1e3,label='Fitted ODE');axs[0].set(xlabel='Time (us)',ylabel='Tag current (mA)');axs[0].legend(fontsize=8)
    axs[1].hist((pred-obs)/noise,bins=25);axs[1].set(xlabel='Residual / noise std',ylabel='Count');fig.tight_layout();save('parameter_fit')
    distances=np.linspace(.02,.12,35);angles=np.linspace(0,np.pi/2,35);Ms=np.array([[mutual_inductance(d=d,tilt=a,n=100)*1e9 for d in distances] for a in angles])
    fig,ax=plt.subplots(figsize=(8,4));im=ax.imshow(Ms,origin='lower',extent=[2,12,0,90],aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax,label='Mutual inductance (nH)');ax.set(xlabel='Coil distance (cm)',ylabel='Tilt (degrees)',title='Geometric 3D coupling');save('coupling_map')
    fig=plt.figure(figsize=(8,5));ax=fig.add_subplot(projection='3d')
    for radius,center,angle,label in [(.04,(0,0,0),0,'Reader'),(.015,(0,0,.04),np.pi/6,'Tag')]:
        r,_=loop(radius,center,angle);ax.plot(*r.T,label=label)
    ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title='3D coil geometry');ax.legend();save('coils_3d')
    metrics['environment']={'python':platform.python_version(),'numpy':np.__version__,'seed':20261029}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2,default=lambda v:bool(v) if isinstance(v,(bool,np.bool_)) else float(v)))
    (ROOT/'data/provenance.json').write_text(json.dumps(provenance,indent=2))
    render(ROOT)
    print(json.dumps(metrics,indent=2,default=str))
if __name__=='__main__':main()
