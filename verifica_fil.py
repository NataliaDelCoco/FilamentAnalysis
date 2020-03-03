# verifica se tendencia do filamento é real

import numpy as np
import pandas as pd
import sys
import itertools as IT
import scipy.spatial as spatial
import astropy.units as un
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70 * un.km / un.s / un.Mpc, Om0=0.3)
import os
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib
import skymapper as skm
import scipy as scp
import itertools

#USO:
# python fil_length.py  ra_dec_fil  ra_dec_gals_original  conv_factor_kpc2arcsec  autput_file

#==================================
# FUNCOES 
#==================================
def arc_length(x, y):
    npts = np.size(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)

    return arc


def deg2Mpc(conv,comp_deg):
  comp_mpc = comp_deg*conv*3600/1000
  return comp_mpc

def Mpc2deg(conv,comp_mpc):
  comp_deg=comp_mpc/(conv*3600/1000)
  return comp_deg

def ptos_within(ptos_orig, ptos_fil, dist):
    # build the KDTree using the *smaller* points array
    tree = spatial.cKDTree(ptos_fil)
    groups = tree.query_ball_point(ptos_orig, dist)
    indices = np.unique([i for i, grp in enumerate(groups) if np.size(grp)])
    return indices

def area_ellipse(conv, ra, dec):
  r_ra = max(ra) - min(ra)
  r_dec = max(dec) - min(dec)

  r_ra_mpc = deg2Mpc(conv, r_ra)
  r_dec_mpc = deg2Mpc(conv, r_dec)

  area=np.pi*r_ra_mpc*r_dec_mpc
  return area #mpc^2

#normaliza
def normm1(xarr):
  minx=min(xarr)
  maxx=max(xarr)
  nrm=2*((xarr-minx)/(maxx-minx))-1
  return nrm

def closest(ptora, ptodec, array):
  dist=[]
  for item in array:
    d=np.sqrt((ptora-item[0])**2 + (ptodec - item[1])**2)
    dist.append(d)
  d_aux=np.sort(dist)
  menor=d_aux[0]
  arg=dist.index(menor)
  return(arg)

def roda(ra0,dec0, ra_fil, dec_fil, ang):
  ra_gira=(ra_fil - ra0)*np.cos(ang) + (dec_fil - dec0)*np.sin(ang) +ra0
  dec_gira=-(ra_fil - ra0)*np.sin(ang) + (dec_fil - dec0)*np.cos(ang) + dec0
  return ra_gira, dec_gira
#================================
# ENTRADAS
#============================
h=0.7 #H=70
in_data=sys.argv[1] #filamento detectado
in_data_orig=sys.argv[2] #galaxias originais
conv=float(sys.argv[3]) #conversao de kpc2arcsec
out_name=sys.argv[4] #sem extensao
ra00=float(sys.argv[7]) #posicao aglr central
dec00=float(sys.argv[8])
cl_indt='realcl.txt'
fil_indt='realfil.txt'
blobs='blobs.txt'

#-----------------------------------
data=pd.read_csv(in_data, delimiter=',')
#ordena dados pelo ra
filra=data['col1']
fildec=data['col2']
aux=np.asarray((filra,fildec)).T
ptos_fil=aux[aux[:,0].argsort()]

ra=ptos_fil[:,0]
dec=ptos_fil[:,1]

idx0 = closest(ra00,dec00, ptos_fil)

ra0=ra[idx0]
dec0=dec[idx0]


#----------------------------------
orig=pd.read_csv(in_data_orig, delimiter=',')
ptos_orig = orig.as_matrix(columns=orig.columns[0:2]) #ra,dec
r = np.asarray(orig['r']) #mag r
g = np.asarray(orig['g']) #mag g
i = np.asarray(orig['i']) #mag i
z = np.asarray(orig['z']) #redshift

#--------LE ENTRADAS - clusters --------------
cl_dt=pd.read_csv(cl_indt, delimiter=',', header=None)
clra=cl_dt[0]
cldec=cl_dt[1]
ptos_cl_arc=np.asarray((clra,cldec)).T
clra=np.asarray(clra)
cldec=np.asarray(cldec)

#--------LE ENTRADAS - filamentos--------------
fil_dt=pd.read_csv(fil_indt, delimiter=',', header=None)
filra=fil_dt[0]
fildec=fil_dt[1]
ptos_fil_arc=np.asarray((filra,fildec)).T
filra=np.asarray(filra)
fildec=np.asarray(fildec)

#--------LE ENTRADAS - blobs --------------
blb_dt=pd.read_csv(blobs, delimiter=',', header=None)
blbra=blb_dt[0]
blbdec=blb_dt[1]
ptos_blb=np.asarray((blbra,blbdec)).T
blbra=np.asarray(blbra)
blbdec=np.asarray(blbdec)

#regiao de 1.5 Mpc ao redor dos blobs
r_excl=Mpc2deg(conv,1.5) #arcmin

#exclui galaxias ao redor de 1.5 mpc dos blobs cetectados
idx_out=[]
ptos_clean=[]
r_clean=[]
g_clean=[]
i_clean=[]
z_clean=[]
for f in range(np.size(blbra)):
  ra_excl, dec_excl =ptos_blb[f][0], ptos_blb[f][1]
  for q in range(int(np.size(ptos_orig)/2)):
    dist = np.sqrt((ra_excl-ptos_orig[q][0])**2 + (dec_excl - ptos_orig[q][1])**2)
    if (dist <= r_excl):
      idx_out.append(q)
#remos indices repetidos:
idx_out = list(dict.fromkeys(idx_out))
for q in range(int(np.size(ptos_orig)/2)):
  if q not in idx_out:
    ptos_clean.append([ptos_orig[q][0],ptos_orig[q][1]])
    r_clean.append(r[q])
    g_clean.append(g[q])
    i_clean.append(i[q])
    z_clean.append(z[q])

#============================
# MAIN
#============================
#densidade do campo
area_field = area_ellipse(conv, np.asarray(orig['ra']), np.asarray(orig['dec']))/(h*h)
densi_field = np.size(orig['ra'])/area_field #num_gal/area (mpc^2)

#luminosidade do filamento----------------------------------------------------
#distancia luminosidade
Ld=np.asarray(cosmo.luminosity_distance(z).to(un.parsec))
#banda r
Mr_sun = 4.67
Mr = np.array(r) + 5 - 5*np.log10(Ld)
Lr = 10**(-(Mr-Mr_sun)/2.5)
#banda g
Mg_sun = 5.36
Mg = np.array(g) + 5 - 5*np.log10(Ld)
Lg = 10**(-(Mg-Mg_sun)/2.5)
#banda i
Mi_sun = 4.48
Mi = np.array(i) + 5 - 5*np.log10(Ld)
Li = 10**(-(Mi-Mi_sun)/2.5)

#===============================================
# fatias do filamento
#==============================================
#mesma analise do que a superior
#so que dividindo do filamento em partes

# faz o processo varias vezes, girando o filamento ao redor do aglr central
angs=[0,np.pi/4,np.pi/2, np.pi, np.pi*(3/2),np.pi*(7/4)]

compri_th=[]
compri_mpc_th=[]
densi_th=[]
in_idx_th=[]
in_th=[]
Lr_th=[]
Lg_th=[]
Li_th=[]
r_th=[]
g_th=[]
i_th=[]
gr_th=[]
ri_th=[]
gi_th=[]
r_th_er=[]
g_th_er=[]
i_th_er=[]
gr_th_er=[]
ri_th_er=[]
gi_th_er=[]
densi_rlt_th=[]
ptos_th=[]
densi_th_fil=[]
densi_rlt_th_fil=[]


for theta in angs:
  ra_th,dec_th=roda(ra0,dec0,ra,dec,theta)

  n_slc=20
  ra_slc=np.array_split(ra_th,n_slc)
  dec_slc=np.array_split(dec_th,n_slc)

  compri_slc=[]
  compri_mpc_slc=[]
  densi_slc=[]
  in_idx_slc=[]
  in_slc=[]
  Lr_slc=[]
  Lg_slc=[]
  Li_slc=[]
  r_slc=[]
  g_slc=[]
  i_slc=[]
  gr_slc=[]
  ri_slc=[]
  gi_slc=[]
  r_slc_er=[]
  g_slc_er=[]
  i_slc_er=[]
  gr_slc_er=[]
  ri_slc_er=[]
  gi_slc_er=[]
  densi_rlt_slc=[]
  ptos_slc=[]
  densi_slc_fil=[]
  densi_rlt_slc_fil=[]


#faz as contas para cada fatia
  for f in range(n_slc):
    ptos_slc_fil = np.column_stack((ra_slc[f],dec_slc[f]))
    #calcula valores importantes----------------------------------------
    #fita subarray
    p1=scp.interpolate.interp1d(ra_slc[f],dec_slc[f])
    xp = np.linspace(min(ra_slc[f]), max(ra_slc[f]),100)
    aj = p1(xp)

    
    compri_slc.append(arc_length(xp, aj))
    compri_mpc_slc.append(deg2Mpc(conv,compri_slc[f])/h)
    densi_slc_fil.append(np.size(ra_slc[f])/compri_mpc_slc[f]) #numero de galaxias por mpc
    densi_rlt_slc_fil.append(densi_slc_fil[f]/densi_field)
    #numero de galaxias mais proximas do que 1Mpc/h ao eixo do filamento-----------
    M=1/h
    d=Mpc2deg(conv,M)
    in_idx_slc.append(ptos_within(ptos_clean, ptos_slc_fil, d))
    in_slc.append(np.size(in_idx_slc[f]))
    densi_slc.append(in_slc[f]/(compri_mpc_slc[f]*M))
    densi_rlt_slc.append(densi_slc[f]/densi_field)

    #Luminosidade, magnitude e cor do filamento -------------------------------------------
    #1 Mpc (soma da luminosidade de todas as galáxias nesses range)
    Lr_aux = 0
    Lg_aux = 0
    Li_aux = 0
    r_aux=[]
    g_aux=[]
    i_aux=[]
    aux_ptos_slc=[]
    for item in in_idx_slc[f]:
      Lr_aux = Lr_aux + Lr[item] #unidades L_sun
      Lg_aux = Lg_aux + Lg[item]
      Li_aux = Li_aux + Li[item]
      r_aux.append(r[item])
      g_aux.append(g[item])
      i_aux.append(i[item])
      aux_ptos_slc.append([ptos_clean[item][0],ptos_clean[item][1]])

    if (np.size(in_idx_slc[f]) > 0):
      ptos_slc.append(aux_ptos_slc)
      Lr_slc.append(Lr_aux/1e10)
      Lg_slc.append(Lg_aux/1e10)
      Li_slc.append(Li_aux/1e10)
      r_slc.append(sum(r_aux)/in_slc[f])
      g_slc.append(sum(g_aux)/in_slc[f])
      i_slc.append(sum(i_aux)/in_slc[f])
      gr_slc.append(g_slc[f] - r_slc[f])
      ri_slc.append(r_slc[f] - i_slc[f])
      gi_slc.append(g_slc[f] - i_slc[f])

      r_slc_er.append(np.std(r_aux)/np.sqrt(in_slc[f]))
      g_slc_er.append(np.std(g_aux)/np.sqrt(in_slc[f]))
      i_slc_er.append(np.std(i_aux)/np.sqrt(in_slc[f]))
      gr_slc_er.append(np.sqrt( g_slc_er[f]**2 + r_slc_er[f]**2 ))
      ri_slc_er.append(np.sqrt( r_slc_er[f]**2 + i_slc_er[f]**2 ))
      gi_slc_er.append(np.sqrt( g_slc_er[f]**2 + i_slc_er[f]**2 ))
    else:
      ptos_slc.append([999999,999999])
      r_slc.append(999999)
      g_slc.append(999999)
      i_slc.append(999999)
      gr_slc.append(999999)
      ri_slc.append(999999)
      gi_slc.append(999999)
      Lr_slc.append(999999)
      Lg_slc.append(999999)
      Li_slc.append(999999)
      r_slc_er.append(999999)
      g_slc_er.append(999999)
      i_slc_er.append(999999)
      gr_slc_er.append(999999)
      ri_slc_er.append(999999)
      gi_slc_er.append(999999)

  compri_th.append(compri_slc)
  compri_mpc_th.append(compri_mpc_slc)
  densi_th.append(densi_slc)
  in_idx_th.append(in_idx_slc)
  in_th.append(in_slc)
  Lr_th.append(Lr_slc)
  Lg_th.append(Lg_slc)
  Li_th.append(Li_slc)
  r_th.append(r_th)
  g_th.append(g_slc)
  i_th.append(i_slc)
  gr_th.append(gr_slc)
  ri_th.append(ri_slc)
  gi_th.append(gi_slc)
  r_th_er.append(r_slc_er)
  g_th_er.append(g_slc_er)
  i_th_er.append(i_slc_er)
  gr_th_er.append(gr_slc_er)
  ri_th_er.append(ri_slc_er)
  gi_th_er.append(gi_slc_er)
  densi_rlt_th.append(densi_rlt_slc)
  ptos_th.append(ptos_slc)
  densi_th_fil.append(densi_slc_fil)
  densi_rlt_th_fil.append(densi_rlt_slc_fil)


#===============================================
# salva
#==============================================
for i_theta in range(len(angs)):
  out_name_f4=out_name+'_slices'+angs[i_theta]+'.csv'

  compri_slc=compri_th[i_theta]
  compri_mpc_slc=compri_mpc_th[i_theta]
  in_slc=in_th[i_theta]
  densi_rlt_slc=densi_rlt_th[i_theta]
  Lr_slc=Lr_th[i_theta]
  Li_slc=Li_th[i_theta]
  Lg_slc=Lg_th[i_theta]
  r_slc=r_th[i_theta]
  r_slc_er=r_th_er[i_theta]
  i_slc=i_th[i_theta]
  i_slc_er=i_th_er[i_theta]
  g_slc=g_th[i_theta]
  g_slc_er=g_th_er[i_theta]
  gr_slc=gr_th[i_theta]
  gr_slc_er=gr_th_er[i_theta]
  ri_slc=ri_th[i_theta]
  ri_slc_er=ri_th_er[i_theta]
  gi_slc=gi_th[i_theta]
  gi_slc_er=gi_th_er[i_theta]

  col0=[]
  for q in range(len(compri_slc)):
    col0.append(q+1)

  salva=pd.DataFrame(columns=['Group', 'Length (deg)', 'Length (Mpc)','Ng','Relative Density', 'Lum_r', 'Lum_g', 'Lum_i', 'r', 'r_er','g', 'g_er', 'i', 'i_er', 'gr', 'gr_er', 'ri', 'ri_er', 'gr', 'gr_er'])
  for q in range(len(densi_slc)):
    salva.loc[q] = col0[q], compri_slc[q], compri_mpc_slc[q], in_slc[q], densi_rlt_slc[q], Lr_slc[q], Lg_slc[q], Li_slc[q], r_slc[q], r_slc_er[q], g_slc[q], g_slc_er[q], i_slc[q], i_slc_er[q], gr_slc[q], gr_slc_er[q], ri_slc[q], ri_slc_er[q], gi_slc[q], gi_slc_er[q]
  salva.to_csv(out_name_f4)


#====================================================================
# PLOTA
#====================================================================
#Acha maximos e minimos pras normalizacoes das cores
mean=[]
stdv=[]
median=[]
for l in range(np.size(angs)):
  mean.append(np.mean(gr_th[l]))
  stdv.append(np.std(gr_th[l]))
  median.append(np.median(gr_th[l]))

max_slc=np.amax(gr_th)
min_slc=np.amin(gr_th)


#-- plot fatias do filamento ----------
jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=min_slc, vmax=max_slc)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
sz=[20,24]*int(n_slc/2)
markers=['v','^']*int(n_slc/2)

crit = skm.stdDistortion
proj = skm.EqualEarth.optimize(ra, dec, crit=crit)
map = skm.Map(proj)
sep=0.5
map.grid(sep=sep)
len = 10
size = 100*np.random.rand(len)
aux_clean=np.array(ptos_clean).T
map.scatter(aux_clean[0], aux_clean[1],s=3, edgecolor='k', facecolor='k', alpha=0.2)
map.focus(aux_clean[0], aux_clean[1])

map.scatter(filra,fildec, s=3,edgecolor='gray', facecolor='gray', marker='.')
map.scatter(clra,cldec, s=30,edgecolor='k', facecolor='k', marker='*')

alphas=[0.2,0.4,0.4,0.4,0.4,0.4]

for i_th in range(np.size(angs)):
  cores=scalarMap.to_rgba(gr_th[i_th])

  ra_th,dec_th=roda(ra0,dec0,ra,dec,angs[i_th])
  ra_slc=np.array_split(ra_th,n_slc)
  dec_slc=np.array_split(dec_th,n_slc)
  ptos_slc=ptos_th[i_th]


  for f in range(n_slc):
    pp=f+1
    map.scatter(ra_slc[f], dec_slc[f], c=cores[f], marker='.', s=6,alpha=alphas[i_th])
    # aux_slc=np.array(ptos_slc[f]).T
    # map.scatter(aux_slc[0], aux_slc[1], marker=markers[f], c=cores[f], alpha=alphas[i_th], s=sz[f], edgecolors=cores[f], label='part %.f' % pp)

for i_th in range(np.size(angs)):
  cores=scalarMap.to_rgba(mean[i_th])

  ra_th,dec_th=roda(ra0,dec0,ra,dec,angs[i_th])
  map.scatter(ra_th, dec_th, c=cores, marker='.', s=6,alpha=alphas[i_th])
    

cb = map.colorbar(scalarMap, cb_label="$g-r$")

print("qual o deslocamento da escala?")
des_x=float(input("horizontal: "))
des_y=float(input("vertical: "))

#escala
scl_deg=Mpc2deg(conv,5)
ra_scl=np.linspace(max(ra)-des_x-scl_deg/2,max(ra)-des_x+scl_deg/2,100)
dec_scl=np.array([min(dec)+des_y]*100)
map.plot(ra_scl,dec_scl, color='k')

input("Pronto?")


outp1=out_name+'_slices.png'
map.savefig(outp1)

#============================================================

#cor
gr_o=g-r


gr=[]
ptos_menor=[]
for l in range(np.size(gr_o)):
  if gr_o[l] > 1.5:
   gr.append(gr_o[l])
   ptos_menor.append(ptos_orig[l].tolist())



max_gr=np.amax(gr)
min_gr=np.amin(gr)

jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=min_gr, vmax=max_gr)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
# sz=[20,24]*int(n_slc/2)
# markers=['v','^']*int(n_slc/2)


cores=scalarMap.to_rgba(gr)

crit = skm.stdDistortion
proj = skm.EqualEarth.optimize(ra, dec, crit=crit)
map = skm.Map(proj)
sep=0.5
map.grid(sep=sep)
len = 10
size = 100*np.random.rand(len)

aux_clean=np.array(ptos_menor).T
# aux_clean=np.array(ptos_orig).T
map.scatter(aux_clean[0], aux_clean[1],s=50, color=cores)
map.focus(aux_clean[0], aux_clean[1])
cb = map.colorbar(scalarMap, cb_label="$g-r$")

map.scatter(filra,fildec, s=3,edgecolor='gray', facecolor='gray', marker='.', alpha=0.5)
map.scatter(clra,cldec, s=30,edgecolor='k', facecolor='k', marker='*', alpha=0.5)




ptos_menor=np.array(ptos_menor)

ra_o=ptos_orig.T[0]
dec_o=ptos_orig.T[1]
gr_o=g-r

max_slc=np.amax(gr_o)
min_slc=np.amin(gr_o)
jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=min_slc, vmax=max_slc)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
cores=scalarMap.to_rgba(gr_o)

crit = skm.stdDistortion
proj = skm.EqualEarth.optimize(ra_o, dec_o, crit=crit)
map = skm.Map(proj)
sep=0.5
map.grid(sep=sep)
len = 10
size = 100*np.random.rand(len)
map.focus(ra_o,dec_o)

map.scatter(ra_o,dec_o,color='r', s=5)


M=5/h
d=Mpc2deg(conv,M)

in_idx0=ptos_within(ptos_orig, ptos_fil, d)
ra_clean=np.array(ptos_clean).T[0]
dec_clean=np.array(ptos_clean).T[1]
r_aux=[]
g_aux=[]
gal_fil_ra=[]
gal_fil_dec=[]
for item in in_idx0:
  r_aux.append(r[item])
  g_aux.append(g[item])
  gal_fil_ra.append(ra_o[item])
  gal_fil_dec.append(dec_o[item])



gr_slc0=np.array(g_aux)-np.array(r_aux)

max_slc=np.amax(gr_slc0)
min_slc=np.amin(gr_slc0)
jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=min_slc, vmax=max_slc)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
cores=scalarMap.to_rgba(gr_o)

crit = skm.stdDistortion
proj = skm.EqualEarth.optimize(ra_o, dec_o, crit=crit)
map = skm.Map(proj)
sep=0.5
map.grid(sep=sep)
len = 10
size = 100*np.random.rand(len)
map.focus(ra_o,dec_o)


cores=scalarMap.to_rgba(gr_slc0)
map.scatter(np.array(gal_fil_ra),np.array(gal_fil_dec),color=cores, s=5)
cb = map.colorbar(scalarMap, cb_label="$g-r$")