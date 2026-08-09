import meshio,numpy as np,xml.etree.ElementTree as ET,os,sys
rundir,out = sys.argv[1],sys.argv[2]
root=ET.parse(os.path.join(rundir,'pp-fluid.pvd')).getroot()
ds=sorted([(float(d.attrib['timestep']),d.attrib['file']) for d in root.iter('DataSet')])
f=os.path.join(rundir,ds[-1][1]); f=f[:-5]+'-0.vtu' if f.endswith('.pvtu') else f
m=meshio.read(f); pts=m.points; p=m.point_data['pressure']
key=np.round(pts[:,:2],10); _,idx=np.unique(key,axis=0,return_index=True)
pts,p=pts[idx],p[idx]
sel=np.abs(pts[:,1]-0.2)<1e-9
x=pts[sel,0]; pv=p[sel]; o=np.argsort(x)
np.save(out,np.vstack([x[o],pv[o]]))
print(f'{out}: p(0)={pv[o][0]:.3f} p(L)={pv[o][-1]:.3f} integral={np.trapz(pv[o],x[o]):.3f}')
