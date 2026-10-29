"""Render time-dependent Ey slice from the saved full 3D solution."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from matplotlib import colormaps

def render(root):
    f=np.load(root/'results/fields_3d.npz');E=f['E'];frames=[]
    lim=float(abs(E[:,1,:,:,12]).max());cmap=colormaps['RdBu_r']
    for i,t in enumerate(f['times']):
        a=(E[i,1,:,:,12].T+lim)/(2*lim)
        im=Image.fromarray((cmap(a)*255).astype('uint8')).convert('RGB').resize((480,480),Image.Resampling.NEAREST)
        canvas=Image.new('RGB',(480,515),'white');canvas.paste(im,(0,35))
        ImageDraw.Draw(canvas).text((10,10),f'Maxwell 3D: Ey at z=0.5, t={t:.3f} (normalized)',fill='black');frames.append(canvas)
    frames[0].save(root/'results/maxwell_evolution.gif',save_all=True,append_images=frames[1:],duration=180,loop=0)
if __name__=='__main__':render(Path(__file__).resolve().parents[1])
