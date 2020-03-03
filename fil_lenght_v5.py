# calcula comprimento d e aglomerados

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
from scipy import stats
from matplotlib.ticker import FuncFormatter
# from cor_k import calc_kcor
from check_distribution import distribution
from anal_cor import anal_cor_fil
from DistInClust import GalsinCl_quadrado
from DistInClust import GalsNotinCl_quadrado
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

def distancia_Mpc(ptora, ptodec, array, conv):
  dist=[]
  for item in array:
    d=np.sqrt((ptora-item[0])**2 + (ptodec - item[1])**2)
    dist.append(d)
  d_aux=np.sort(dist)
  menor=d_aux[0]
  menor_Mpc=menor*conv*3600/1000

  return menor, menor_Mpc

def printDivisors(n) : 
    i = 1
    while i <= n : 
        if (n % i==0) : 
            print(i), 
        i = i + 1

def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

def abs_mag(ap_mag, z, zErr, K_corr): #Michael R. Blanton and Sam Roweis, 2006
  #valor de M
  print('Cálculo da Mag_absoluta')
  M=[]
  for d in range(len(z)):
    DL=(cosmo.luminosity_distance(z[d]).value)*1e6 #1e6 para passar de Mpc para pc
    DM=5*np.log10(DL/10)
    M_aux = ap_mag[d] - DM - K_corr[d] 

    M.append(M_aux)
  Merr = np.zeros(len(M))

  return M, Merr

def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '%.2f' % (x)

#================================
# ENTRADAS
#================================

conv=float(sys.argv[1]) #conversao de kpc2arcsec
# in_data=sys.argv[1] #filamento detectado
# in_data_orig=sys.argv[2] #galaxias originais
# conv=float(sys.argv[3]) #conversao de kpc2arcsec
# out_name=sys.argv[4] #sem extensao
# excl_cls=sys.argv[5] #posicao dos aglomerados que estao no fil que serao removidos
# fil_init = sys.argv[5] #ponto a parti do qual buscaremos os mais proximos (inicio do filamento)
h=0.7 #H=70
in_data='f1.csv'
in_data_orig='data_med_mapa_n_v2.csv'
out_name='f1'
excl_cls='real_cls_in_fils.csv'
fil_init='f1_init.csv'
#-----------------------------------

fint=pd.read_csv(fil_init, header=None)
ra00=float(fint[0])
dec00=float(fint[1])
#----------------------------------

data=pd.read_csv(in_data, delimiter=',')
#ordena dados pelo ra
filra=data['col1']
fildec=data['col2']
aux=np.asarray((filra,fildec)).T
ptos_fil=aux[aux[:,0].argsort()]

ra=ptos_fil[:,0]
dec=ptos_fil[:,1]

aux_ptos_fil=ptos_fil
novo_ptos_fil=[]

idx0 = closest(ra00,dec00, aux_ptos_fil)

novo_ptos_fil.append(ptos_fil[idx0].tolist())
aux_ptos_fil = np.delete(aux_ptos_fil,idx0, axis=0)

count=0
while (count < (np.size(ra) -1)):
# while (count < (992)):

  aux=novo_ptos_fil[-1]
  ptora=aux[0]
  ptodec=aux[1]
  ind_closest = closest(ptora,ptodec, aux_ptos_fil)
  novo_ptos_fil.append(aux_ptos_fil[ind_closest].tolist())
  aux_ptos_fil = np.delete(aux_ptos_fil, ind_closest,axis=0)

  count= count+1
novo_ptos_fil = np.asarray(novo_ptos_fil)
ra=novo_ptos_fil[:,0]
dec=novo_ptos_fil[:,1]


#----------------------------------
orig=pd.read_csv(in_data_orig, delimiter=',')
ptos_orig = orig.as_matrix(columns=orig.columns[0:2]) #ra,dec
r = np.asarray(orig['dered_r']) #mag r_dered
g = np.asarray(orig['dered_g']) #mag g_dered
i = np.asarray(orig['dered_i']) #mag i_dered
Kcorr_g=np.asarray(orig['Kcorr_g'])
Kcorr_i=np.asarray(orig['Kcorr_i'])
Kcorr_r=np.asarray(orig['Kcorr_r'])
z = np.asarray(orig['z']) #redshift
zErr = np.asarray(orig['z_err'])

#----------------------------------
excl=pd.read_csv(excl_cls, delimiter=',')
ptos_excl = excl.as_matrix(columns=excl.columns[1:3]) #ra,dec
r_excl = np.asarray(excl['R(arcmin)'])/60 #GRAUS ptos dentro do raio R serao excluidos
z_excl = np.asarray(excl['Redshift']) #redshift
ra_cls = np.asarray(excl['RA'])
dec_cls = np.asarray(excl['DEC'])


#======================================================
# MAIN
#=====================================================
# correcao k das cores
# cor g
# kg=[]
# for d in range(len(z)):
#   aux_cor= g[d] - r[d]
#   corr=calc_kcor('g', z[d], 'g - r', aux_cor)
#   kg.append(corr)

# ki=[]
# for q in range(len(z)):
#   aux_cor=g[q]-i[q]
#   corr=calc_kcor('i', z[q], 'g - i', aux_cor)
#   ki.append(corr)

# ki_ui=[]
# for q in range(len(z)):
#   aux_cor=u[q]-i[q]
#   corr=calc_kcor('i', z[q], 'u - i', aux_cor)
#   ki_ui.append(corr)

# kg_gi=[]
# for q in range(len(z)):
#   aux_cor=g[q]-i[q]
#   corr=calc_kcor('g', z[q], 'g - i', aux_cor)
#   kg_gi.append(corr)

# bins=[500,500]
# xyrange = [[0,max(z)],[-0.5,0.5]]
# ydata=ki_ui
# yd_sdss=dt['kcorrI']
# ylab='ki (u-i)'
# fig, axes = plt.subplots(2, sharex=True, sharey=True, gridspec_kw={'hspace': 0})
# hh, locx, locy = scipy.histogram2d(z, ydata, range=xyrange, bins=bins)
# dd=1
# for ax in axes.flat:
#   if dd==1:
#     im=ax.imshow(np.flipud(hh.T),cmap='nipy_spectral',extent=np.array(xyrange).flatten(), interpolation='none', origin='upper', aspect='auto')
#   else:
#     im=ax.imshow(np.flipud(hh.T),cmap='nipy_spectral',extent=np.array(xyrange).flatten(), interpolation='none', origin='upper', aspect='auto')
#     ax.scatter(z, yd_sdss, s=0.5, color='white', alpha=0.2)
#   dd +=1
# plt.xlim((0,max(z)))
# plt.ylim((-0.5,0.5))
# plt.xlabel('z')
# fig.text(0.04, 0.5, ylab, va='center', rotation='vertical')
# cbar_ax = fig.add_axes([0.91, 0.12, 0.02, 0.75])
# fig.colorbar(im, cax=cbar_ax)


# K_g_gr_corr=[]
# for q in range(len(ptos_orig)):
#   aux_cor=g[q]-r[q]
#   corr=calc_kcor('g', z[q], 'g - r', aux_cor)
#   K_g_gr_corr.append(corr)

# # cor r
# K_r_gr_corr=[]
# for q in range(len(ptos_orig)):
#   aux_cor=g[q]-r[q]
#   corr=calc_kcor('r', z[q], 'g - r', aux_cor)
#   K_r_gr_corr.append(corr)

# # cor i
# K_i_gi_corr=[]
# for q in range(len(ptos_orig)):
#   aux_cor=g[q]-i[q]
#   corr=calc_kcor('i', z[q], 'g - i', aux_cor)
#   K_i_gi_corr.append(corr)

# # Magnitude absoluta corrigida das galaxias
# Mg_o,MgErr = abs_mag(g, z, zErr, K_g_gr_corr)
# Mi_o,MiErr = abs_mag(i, z, zErr, K_i_gi_corr)
# Mr_o,MrErr = abs_mag(r, z, zErr, K_r_gr_corr)
Mg_o,MgErr=abs_mag(g, z, zErr, Kcorr_g)
Mi_o,MiErr=abs_mag(i, z, zErr, Kcorr_i)
Mr_o,MrErr=abs_mag(r, z, zErr, Kcorr_r)
z_o = z.tolist()
zErr_o = zErr.tolist()
ptos_orig_o=ptos_orig.tolist()

gi=np.array(Mg_o)-np.array(Mi_o)
plt.scatter(Mg_o,gi,s=1, color='b')
plt.xlabel('Mg')
plt.ylabel('g-i')
# plt.show()

print('')
# Mg_max=np.float(input('Qual o corte superior de Mg? '))
# Mg_min=np.float(input('Qual o corte inferior de Mg? '))
# gi_max=np.float(input('Qual o corte superior de (g-i)? '))
# gi_min=np.float(input('Qual o corte inferior de (g-i)? '))
Mg_max=-17.5
Mg_min=-23.5
gi_max=2
gi_min=0

aa=[]
for d in range(len(z_o)):
  if Mg_o[d] < Mg_min or Mg_o[d]>Mg_max or gi[d] < gi_min or gi[d] > gi_max:
    aa.append(d)

  # if Mr[d] < -21:
  #   aa.append(d)
  # if Mi[d] < -21:
  #   aa.append(d)
  # if gr > 2 or gr < -2:
  #   aa.append(d)
Mg=[]
Mi=[]
Mr=[]
z=[]
zErr=[]
ptos_orig=[]
idx_apaga = (np.unique(np.sort(aa))).tolist()
for n in range(len(z_o)):
  if n not in idx_apaga:
    Mg.append(Mg_o[n])
    Mi.append(Mi_o[n])
    Mr.append(Mr_o[n])
    z.append(z_o[n])
    zErr.append(zErr_o[n])
    ptos_orig.append(ptos_orig_o[n])

gi=np.array(Mg)-np.array(Mi)
plt.scatter(Mg,gi,s=1,color='orange')
plt.xlabel('Mg')
plt.ylabel('g-i')
plt.show()
input('continua?')

#cria sample limpo de clusters----------
idx_out0=[]
idx_agl=[]
dist_agl=[]
dist_filagl=[]
dist_galfil_agl=[]
idx_GalsInCl=[]
for q in range(len(r_excl)):
  aux_out,aux_agl, aux_dist, aux_dist2, aux_dist3=GalsinCl_quadrado(r_excl[q], ra_cls[q], dec_cls[q], ptos_fil,ptos_orig, conv)
  idx_out0.append(aux_out)
  idx_agl.append(aux_agl)
  dist_agl.append(aux_dist)
  dist_filagl.append(aux_dist2)
  dist_galfil_agl.append(aux_dist3)


idx_out=[]
for n in range(len(ptos_orig)):
  tira=0
  for q in range(len(r_excl)):
    if np.isin(n,idx_agl[q])==False:
      tira +=1
  if tira == len(r_excl):
    idx_out.append(n)



ptos_clean=[]
Mr_clean=[]
Mg_clean=[]
Mi_clean=[]
z_clean=[]
zErr_clean=[]
#remos indices repetidos:
idx_out = list(dict.fromkeys(idx_out))
for q in range(len(ptos_orig)):
  if q in idx_out:
    ptos_clean.append([ptos_orig[q][0],ptos_orig[q][1]])
    Mr_clean.append(Mr[q])
    Mg_clean.append(Mg[q])
    Mi_clean.append(Mi[q])
    z_clean.append(z[q])
    zErr_clean.append(zErr[q])


#divide o array inicial em subarrays
#IGUAL PARA AS DUAS PARTES
n_subs=30
ra_subs=np.array_split(ra,n_subs)
dec_subs=np.array_split(dec,n_subs)


#fita cada um dos subarrays (interp)
f_subs=[]
xp_subs=[]
aj_subs=[]
for q in range(n_subs):
  f_subs.append(scp.interpolate.interp1d(ra_subs[q],dec_subs[q]))
  xp_subs.append(np.linspace(min(ra_subs[q]), max(ra_subs[q]),100))
  aux_f=f_subs[q]
  aj_subs.append(aux_f(xp_subs[q]))

#junta tudo
xp = list(itertools.chain.from_iterable(xp_subs))
aj = list(itertools.chain.from_iterable(aj_subs))


# for q in range(n_subs):
#   plt.plot(ra_subs[q], dec_subs[q], '.', xp_subs[q], aj_subs[q], '-')
# plt.show()
# plt.close()

#calcula valores importantes----------------------------------------
compri = arc_length(xp, aj)
# compri_mpc = deg2Mpc(conv,compri)/h
compri_mpc = deg2Mpc(conv,compri)
densi_fil = len(ra)/compri_mpc #numero de galaxias por mpc

#densidade relativa ------------------------------------------------------
# area_field = area_ellipse(conv, np.asarray(orig['ra']), np.asarray(orig['dec']))/(h*h)
area_field = area_ellipse(conv, np.asarray(orig['ra']), np.asarray(orig['dec']))
densi_field = np.size(orig['ra'])/area_field #num_gal/area (mpc^2)
densi_fil_rlt=densi_fil/densi_field

# #SALVA MANUAL HEADER============================================================

head_name='fil_README.txt'

if (os.path.isfile(head_name) == False):
  head1=('Referent to files with filaments and blobs informations\n')
  head2=('Ng.5, Ng1 = Number of galaxies closer than 0.5Mpc/h and 1Mpc/h to filament axis\n')
  head3=('".5" and "1" refers to galaxies closer than 0.5Mpc/h and 1Mpc/h to filament axis')
  with open(head_name,'w') as t:
    t.write(head1)
    t.write(head2)
    t.write(head3)
  t.close()

# SALVA AS SAIDAS EM FILES

head1=('Length (deg), Length (Mpc), Density, Relative Density' )
val=[round(compri,4), round(compri_mpc,4), round(densi_fil,4), round(densi_fil_rlt,4)]
out_name_f1=out_name+'_comrpimento.txt'
with open(out_name_f1,'w') as t:
  t.write(head1)
  t.write("\n")
  for y in range(len(val)):
    v=str(val[y])
    t.write(v)
    if (y < (len(val)-1)): 
      t.write(", ")
t.close()

#==============================================================================
# ANALISE SEM EXCLUIR GALS DE AGLOMERADOS
#==========================================================================

#===============================================
# GRADIENTE do filamento
#==============================================

dist_from_fil=[0.25,0.5,0.75,1.,1.25,1.5]
in_dist_idx=[]
in_dist=[]
densi_dist=[]
densi_dif_dist=[]
dist_galfil_deg = []
dist_galfil_mpc = []
idx_total=[]
#numero de galaxias mais proximas do que x*Mpc ao eixo do filamento-----------
for q in range(len(dist_from_fil)):
  M=dist_from_fil[q]
  # M=item/h #Mpc/h
  d=Mpc2deg(conv,M)
  aux_idx = ptos_within(ptos_orig, ptos_fil, d)
  in_dist_idx.append(aux_idx)
  aux_in=np.size(aux_idx)
  in_dist.append(aux_in)
  densi_dist.append(aux_in/(compri_mpc*M))
  if (q == 0):
    densi_dif_dist.append(densi_dist[q])
  else:
    densi_dif_dist.append((in_dist[q]-in_dist[q-1])/(compri_mpc*M))
  idx_total = np.concatenate((idx_total, aux_idx))

#TODAS AS GALAXIAS MAIS PROXIMAS QUE x*Mpc/h DO EIXO DO FILAMENTO
idx_total=np.unique(idx_total)
gr_gals=[]
ri_gals=[]
gi_gals=[]
Mg_gals=[]
Mi_gals=[]
Mr_gals=[]

for idx in idx_total:
  aux_deg, aux_mpc=distancia_Mpc(ptos_orig[int(idx)][0], ptos_orig[int(idx)][1], ptos_fil, conv)
  dist_galfil_deg.append(aux_deg)
  dist_galfil_mpc.append(aux_mpc)
  Mg_gals.append(Mg[int(idx)])
  Mi_gals.append(Mi[int(idx)])
  Mr_gals.append(Mr[int(idx)])
  gr_gals.append(Mg[int(idx)] - Mr[int(idx)])
  ri_gals.append(Mr[int(idx)] - Mi[int(idx)])
  gi_gals.append(Mg[int(idx)] - Mi[int(idx)])



#luminosidade do filamento----------------------------------------------------
#banda r
Mr_sun = 4.67
Lr = 10**(-(np.array(Mr)-Mr_sun)/2.5)
#banda g
Mg_sun = 5.36
Lg = 10**(-(np.array(Mg)-Mg_sun)/2.5)
#banda i
Mi_sun = 4.48
Li = 10**(-(np.array(Mi)-Mi_sun)/2.5)

#ate x*Mpc/h (soma da luminosidade de todas as galáxias nesses range)
Lr_dist = []
Lg_dist = []
Li_dist = []

for q in range(len(dist_from_fil)):
  Lr_aux = 0 #unidades L_sun
  Lg_aux = 0
  Li_aux = 0
  if (in_dist[q] > 0):
    for item in in_dist_idx[q]:
      Lr_aux = Lr_aux + Lr[item] #unidades L_sun
      Lg_aux = Lg_aux + Lg[item]
      Li_aux = Li_aux + Li[item]

    Lr_dist.append(Lr_aux/1e10)
    Lg_dist.append(Lg_aux/1e10)
    Li_dist.append(Li_aux/1e10)
  else:
    Lr_dist.append(999999)
    Lg_dist.append(999999)
    Li_dist.append(999999)


#magnitude e cor do filamento *******************************************
Mr_dist=[]
Mg_dist=[]
Mi_dist=[]
Mr_dist_er=[]
Mg_dist_er=[]
Mi_dist_er=[]
ptos_dist=[]

gr_dist=[]
ri_dist=[]
gi_dist=[]
gr_dist_er=[]
ri_dist_er=[]
gi_dist_er=[]


for q in range(len(dist_from_fil)):
  Mr_aux = [] #unidades L_sun
  Mg_aux = []
  Mi_aux = []
  MrErr_aux = [] #unidades L_sun
  MgErr_aux = []
  MiErr_aux = []
  ptos_aux=[]
  if (in_dist[q] > 0):
    for item in in_dist_idx[q]:
      Mr_aux.append(Mr[item]) #unidades L_sun
      Mg_aux.append(Mg[item])
      Mi_aux.append(Mi[item])
      ptos_aux.append([ptos_orig[item][0],ptos_orig[item][1]])

    ptos_dist.append(ptos_aux)
    Mr_dist.append(sum(Mr_aux)/in_dist[q])
    Mg_dist.append(sum(Mg_aux)/in_dist[q])
    Mi_dist.append(sum(Mi_aux)/in_dist[q])

    gr_dist.append(Mg_dist[q]-Mr_dist[q])
    ri_dist.append(Mr_dist[q]-Mi_dist[q])
    gi_dist.append(Mg_dist[q]-Mi_dist[q])

    Mr_dist_er.append(np.std(Mr_aux)/np.sqrt(in_dist[q]))
    Mg_dist_er.append(np.std(Mg_aux)/np.sqrt(in_dist[q]))
    Mi_dist_er.append(np.std(Mi_aux)/np.sqrt(in_dist[q]))
    gr_dist_er.append(np.sqrt( Mg_dist_er[q]**2 + Mr_dist_er[q]**2 ))
    ri_dist_er.append(np.sqrt( Mr_dist_er[q]**2 + Mi_dist_er[q]**2 ))
    gi_dist_er.append(np.sqrt( Mg_dist_er[q]**2 + Mi_dist_er[q]**2 ))


  else:
    ptos_dist.append([999999,999999])
    Mr_dist.append(999999)
    Mg_dist.append(999999)
    Mi_dist.append(999999)
    gr_dist.append(999999)
    ri_dist.append(999999)
    gr_dist.append(999999)


# pras diferencas -----------------------------------------------------------
Mr_dif_dist=[]
Mg_dif_dist=[]
Mi_dif_dist=[]
gr_dif_dist=[]
ri_dif_dist=[]
gi_dif_dist=[]
Mr_dif_dist_er=[]
Mg_dif_dist_er=[]
Mi_dif_dist_er=[]
gr_dif_dist_er=[]
ri_dif_dist_er=[]
gi_dif_dist_er=[]


ptos_dif_dist=[]
in_dif_dist=[]

for q in range(len(dist_from_fil)):
  Mr_aux = [] #unidades L_sun
  Mg_aux = []
  Mi_aux = []
  ptos_aux=[]
  if (in_dist[q] > 0):
    if (q == 0):
      Mr_dif_dist.append(Mr_dist[q])
      Mg_dif_dist.append(Mg_dist[q])
      Mi_dif_dist.append(Mi_dist[q])
      gr_dif_dist.append(gr_dist[q])
      ri_dif_dist.append(ri_dist[q])
      gi_dif_dist.append(gi_dist[q])
      ptos_aux.append(ptos_dist[q])
      in_dif_dist.append(len(Mr_aux))
      ptos_dif_dist.append(ptos_aux)
      Mr_dif_dist_er.append(Mr_dist_er[q])
      Mg_dif_dist_er.append(Mg_dist_er[q])
      Mi_dif_dist_er.append(Mi_dist_er[q])
      gr_dif_dist_er.append(gr_dist_er[q])
      ri_dif_dist_er.append(ri_dist_er[q])
      gi_dif_dist_er.append(gi_dist_er[q])


    else:
      for item in in_dist_idx[q]:
        if item not in in_dist_idx[q-1]:
          Mr_aux.append(Mr[item]) #unidades L_sun
          Mg_aux.append(Mg[item])
          Mi_aux.append(Mi[item])
          ptos_aux.append([ptos_orig[item][0],ptos_orig[item][1]])

      ptos_dif_dist.append(ptos_aux)
      in_dif_dist.append(len(Mr_aux))
      Mr_dif_dist.append(sum(Mr_aux)/in_dif_dist[q])
      Mg_dif_dist.append(sum(Mg_aux)/in_dif_dist[q])
      Mi_dif_dist.append(sum(Mi_aux)/in_dif_dist[q])

      gr_dif_dist.append(Mg_dif_dist[q]-Mr_dif_dist[q])
      ri_dif_dist.append(Mr_dif_dist[q]-Mi_dif_dist[q])
      gi_dif_dist.append(Mg_dif_dist[q]-Mi_dif_dist[q])


      Mr_dif_dist_er.append(np.std(Mr_aux)/np.sqrt(in_dif_dist[q]))
      Mg_dif_dist_er.append(np.std(Mg_aux)/np.sqrt(in_dif_dist[q]))
      Mi_dif_dist_er.append(np.std(Mi_aux)/np.sqrt(in_dif_dist[q]))
      gr_dif_dist_er.append(np.sqrt( Mg_dif_dist_er[q]**2 + Mr_dif_dist_er[q]**2 ))
      ri_dif_dist_er.append(np.sqrt( Mr_dif_dist_er[q]**2 + Mi_dif_dist_er[q]**2 ))
      gi_dif_dist_er.append(np.sqrt( Mg_dif_dist_er[q]**2 + Mi_dif_dist_er[q]**2 ))

  else:
    ptos_dif_dist.append([999999,999999])
    Mr_dif_dist.append(999999)
    Mg_dif_dist.append(999999)
    Mi_dif_dist.append(999999)
    gr_dif_dist.append(999999)
    ri_dif_dist.append(999999)
    gr_dif_dist.append(999999)

#densidade do campo ------------------------------------------------------------------------
#area em (mpc/h)^2
# area_field = area_ellipse(conv, np.asarray(orig['ra']), np.asarray(orig['dec']))/(h*h)
area_field = area_ellipse(conv, np.asarray(orig['ra']), np.asarray(orig['dec']))
densi_field = np.size(orig['ra'])/area_field #num_gal/area (mpc^2)

#densidade relativa ------------------------------------------------------
densi_rlt=[]
for q in range(len(dist_from_fil)):
  densi_rlt.append(densi_dist[q]/densi_field)

densi_dif_rlt=[]
for q in (range(len(dist_from_fil))):
  densi_dif_rlt.append(densi_dif_dist[q]/densi_field)

#===============================================
# fatias do filamento
#==============================================
#mesma analise do que a superior
#so que dividindo do filamento em partes
n_slc=10
ra_slc=np.array_split(ra,n_slc)
dec_slc=np.array_split(dec,n_slc)

compri_slc=[]
compri_mpc_slc=[]
densi_slc=[]
in_idx_slc=[]
in_slc=[]
densi_slc=[]
Lr_slc=[]
Lg_slc=[]
Li_slc=[]

Mr_slc=[]
Mg_slc=[]
Mi_slc=[]
gr_slc=[]
ri_slc=[]
gi_slc=[]
Mr_slc_er=[]
Mg_slc_er=[]
Mi_slc_er=[]
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
  # compri_mpc_slc.append(deg2Mpc(conv,compri_slc[f])/h)
  compri_mpc_slc.append(deg2Mpc(conv,compri_slc[f]))
  densi_slc_fil.append(np.size(ra_slc[f])/compri_mpc_slc[f]) #numero de galaxias por mpc
  densi_rlt_slc_fil.append(densi_slc_fil[f]/densi_field)
  #numero de galaxias mais proximas do que 1Mpc/h ao eixo do filamento-----------
  # M=1/h
  M=1.
  d=Mpc2deg(conv,M)
  in_idx_slc.append(ptos_within(ptos_orig, ptos_slc_fil, d))
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
    r_aux.append(Mr[item])
    g_aux.append(Mg[item])
    i_aux.append(Mi[item])
    aux_ptos_slc.append([ptos_orig[item][0],ptos_orig[item][1]])

  if (np.size(in_idx_slc[f]) > 0):
    ptos_slc.append(aux_ptos_slc)
    Lr_slc.append(Lr_aux/1e10)
    Lg_slc.append(Lg_aux/1e10)
    Li_slc.append(Li_aux/1e10)
    Mr_slc.append(sum(r_aux)/in_slc[f])
    Mg_slc.append(sum(g_aux)/in_slc[f])
    Mi_slc.append(sum(i_aux)/in_slc[f])
    gr_slc.append(Mg_slc[f] - Mr_slc[f])
    ri_slc.append(Mr_slc[f] - Mi_slc[f])
    gi_slc.append(Mg_slc[f] - Mi_slc[f])

    Mr_slc_er.append(np.std(r_aux)/np.sqrt(in_slc[f]))
    Mg_slc_er.append(np.std(g_aux)/np.sqrt(in_slc[f]))
    Mi_slc_er.append(np.std(i_aux)/np.sqrt(in_slc[f]))
    gr_slc_er.append(np.sqrt( Mg_slc_er[f]**2 + Mr_slc_er[f]**2 ))
    ri_slc_er.append(np.sqrt( Mr_slc_er[f]**2 + Mi_slc_er[f]**2 ))
    gi_slc_er.append(np.sqrt( Mg_slc_er[f]**2 + Mi_slc_er[f]**2 ))
  else:
    ptos_slc.append([999999,999999])
    Mr_slc.append(999999)
    Mg_slc.append(999999)
    Mi_slc.append(999999)
    gr_slc.append(999999)
    ri_slc.append(999999)
    gi_slc.append(999999)
    Lr_slc.append(999999)
    Lg_slc.append(999999)
    Li_slc.append(999999)
    Mr_slc_er.append(999999)
    Mg_slc_er.append(999999)
    Mi_slc_er.append(999999)
    gr_slc_er.append(999999)
    ri_slc_er.append(999999)
    gi_slc_er.append(999999)


#===============================================
# salva
#==============================================

out_name_f2=out_name+'_grad.csv'
col0=['0-0.25', '0.25-0.5','0.5-0.75','0.75-1.', '1.-1.25', '1.25-1.5']
salva=pd.DataFrame(columns=['Distance_from _fil (Mpc/h)','Ng', 'Relative Density', 'Lum_r',
      'Lum_g', 'Lum_i', 'Mr', 'Mr_er','Mg', 'Mg_er', 'Mi', 'Mi_er',
      'gr', 'gr_er', 'ri', 'ri_er', 'gi', 'gi_er'])
for q in range(len(densi_dist)):
  salva.loc[q] = dist_from_fil[q], in_dist[q], densi_rlt[q], Lr_dist[q], Lg_dist[q], Li_dist[q], Mr_dist[q], Mr_dist_er[q], Mg_dist[q], Mg_dist_er[q],  Mi_dist[q], Mi_dist_er[q], gr_dist[q], gr_dist_er[q], ri_dist[q], ri_dist_er[q], gi_dist[q], gi_dist_er[q]
for f in range(len(densi_dif_dist)):
  salva.loc[q+f] = col0[f], in_dif_dist[f], densi_dif_rlt[f], 0, 0, 0, Mr_dif_dist[f], Mr_dif_dist_er[f], Mg_dif_dist[f], Mg_dif_dist_er[f], Mi_dif_dist[f], Mi_dif_dist_er[f], gr_dif_dist[f], gr_dif_dist_er[f], ri_dif_dist[f], ri_dif_dist_er[f], gi_dif_dist[f], gi_dif_dist_er[f]
salva.to_csv(out_name_f2)

# SALVA GALAXIAS IDENTIFICADAS COMO MEMBRO
out_name_f3=out_name+'_grad_membergalaxies.csv'
colra=[]
coldec=[]
for q in range(len(densi_rlt)):
  if (q==0):
    aux='ra_0_%s' % dist_from_fil[q]
    colra.append(aux)
    aux='dec_0_%s' % dist_from_fil[q]
    coldec.append(aux)
  else:
    aux='ra_%s_%s' % (dist_from_fil[q-1], dist_from_fil[q])
    colra.append(aux)
    aux='dec_%s_%s' % (dist_from_fil[q-1], dist_from_fil[q])
    coldec.append(aux)

aux_ptos_dist=np.array(ptos_dist[0]).T
dt={colra[0]: aux_ptos_dist[0],coldec[0]: aux_ptos_dist[1]}
mbrs=pd.DataFrame(dt,columns=[colra[0],coldec[0]] )

for q in range(len(densi_rlt)-1):
  aux_ptos_dist=np.array(ptos_dist[q+1]).T
  aux_ptos_ants=np.array(ptos_dist[q]).T
  auxra=[]
  for item in aux_ptos_dist[0]:
    if item not in aux_ptos_ants[0]:
      auxra.append(item)
  auxdec=[]
  for item in aux_ptos_dist[1]:
    if item not in aux_ptos_ants[1]:
      auxdec.append(item)
  mbrs[colra[q+1]]=pd.Series(auxra)
  mbrs[coldec[q+1]]=pd.Series(auxdec)

mbrs.to_csv(out_name_f3)


out_name_f4=out_name+'_slices.csv'

col0=[]
for q in range(len(compri_slc)):
  col0.append(q+1)

salva=pd.DataFrame(columns=['Group', 'Length (deg)', 'Length (Mpc)','Ng','Relative Density', 'Lum_r', 'Lum_g', 'Lum_i', 'Mr', 'Mr_er','Mg', 'Mg_er', 'Mi', 'Mi_er', 'gr', 'gr_er', 'ri', 'ri_er', 'gr', 'gr_er'])
for q in range(len(densi_slc)):
  salva.loc[q] = col0[q], compri_slc[q], compri_mpc_slc[q], in_slc[q], densi_rlt_slc[q], Lr_slc[q], Lg_slc[q], Li_slc[q], Mr_slc[q], Mr_slc_er[q], Mg_slc[q], Mg_slc_er[q], Mi_slc[q], Mi_slc_er[q], gr_slc[q], gr_slc_er[q], ri_slc[q], ri_slc_er[q], gi_slc[q], gi_slc_er[q]
salva.to_csv(out_name_f4)


out_name_f5=out_name+'_slice_membergalaxies.csv'
colra=[]
coldec=[]
for q in range(len(compri_slc)):
  aux='ra_'+'%s' % q
  colra.append(aux)
  aux='dec_'+'%s' % q
  coldec.append(aux)

aux_ptos_slc=np.array(ptos_slc[0]).T
dt={colra[0]: aux_ptos_dist[0],coldec[0]: aux_ptos_dist[1]}
mbrs=pd.DataFrame(dt,columns=[colra[0],coldec[0]] )

for q in range(len(compri_slc)-1):
  aux_ptos=np.array(ptos_slc[q+1]).T
  # aux_ptos_ants=np.array(ptos_slc[q]).T
  auxra=[]
  for item in aux_ptos[0]:
    # if item not in aux_ptos_ants[0]:
    auxra.append(item)
  auxdec=[]
  for item in aux_ptos[1]:
    # if item not in aux_ptos_ants[1]:
    auxdec.append(item)
  mbrs[colra[q+1]]=pd.Series(auxra)
  mbrs[coldec[q+1]]=pd.Series(auxdec)

mbrs.to_csv(out_name_f5)


#==============================================================================
# ANALISE EXCLUINDO GALS DE AGLOMERADOS
#==========================================================================

#===============================================
# GRADIENTE do filamento
#==============================================

dist_e_from_fil=[0.25,0.5,0.75,1.,1.25,1.5]
in_dist_e_idx=[]
in_dist_e=[]
densi_dist_e=[]
densi_dif_dist_e=[]
dist_galfil_deg_e = []
dist_galfil_mpc_e = []
idx_total_e = []

#numero de galaxias mais proximas do que x*Mpc/h ao eixo do filamento-----------
for q in range(len(dist_e_from_fil)):
  M=dist_e_from_fil[q]
  # M=item/h #Mpc/h
  d=Mpc2deg(conv,M)
  aux_idx = ptos_within(ptos_clean, ptos_fil, d)
  in_dist_e_idx.append(aux_idx)
  aux_in=np.size(aux_idx)
  in_dist_e.append(aux_in)
  densi_dist_e.append(aux_in/(compri_mpc*M))

  if (q == 0):
    densi_dif_dist_e.append(densi_dist_e[q])
  else:
    densi_dif_dist_e.append((in_dist_e[q]-in_dist_e[q-1])/(compri_mpc*M))
  idx_total_e = np.concatenate((idx_total_e, aux_idx))

#TODAS AS GALAXIAS MAIS PROXIMAS QUE x*Mpc/h DO EIXO DO FILAMENTO
idx_total_e=np.unique(idx_total_e)
gr_gals_e=[]
ri_gals_e=[]
gi_gals_e=[]
Mg_gals_e=[]
Mi_gals_e=[]
Mr_gals_e=[]
z_gals_e=[]

zErr_gals_e=[]

ptos_clean_fil=[]
for idx in idx_total_e:
  aux_deg, aux_mpc=distancia_Mpc(ptos_clean[int(idx)][0], ptos_clean[int(idx)][1], ptos_fil, conv)
  dist_galfil_deg_e.append(aux_deg)
  dist_galfil_mpc_e.append(aux_mpc)

  z_gals_e.append(z_clean[int(idx)])
  Mg_gals_e.append(Mg_clean[int(idx)])
  Mi_gals_e.append(Mi_clean[int(idx)])
  Mr_gals_e.append(Mr_clean[int(idx)])
  gr_gals_e.append(Mg_clean[int(idx)] - Mr_clean[int(idx)])
  ri_gals_e.append(Mr_clean[int(idx)] - Mi_clean[int(idx)])
  gi_gals_e.append(Mg_clean[int(idx)] - Mi_clean[int(idx)])

  zErr_gals_e.append(zErr_clean[int(idx)])
  ptos_clean_fil.append([ptos_clean[int(idx)][0], ptos_clean[int(idx)][1]])

dist_galcl_deg_e = []
dist_galcl_mpc_e = []
for idx in idx_total_e:
  aux_deg2=[]
  aux_mpc2=[]
  for d in range(len(ptos_excl)):
    ptos_cl=np.array([ptos_excl[d]])
    aux_deg, aux_mpc=distancia_Mpc(ptos_clean[int(idx)][0], ptos_clean[int(idx)][1], ptos_cl, conv)
    aux_deg2.append(aux_deg)
    aux_mpc2.append(aux_mpc)
  dist_galcl_deg_e.append(aux_deg2)
  dist_galcl_mpc_e.append(aux_mpc2)

#luminosidade do filamento----------------------------------------------------
#ja calculada

#ate x*Mpc/h (soma da luminosidade de todas as galáxias nesses range)
Lr_dist_e = []
Lg_dist_e = []
Li_dist_e = []

for q in range(len(dist_e_from_fil)):
  Lr_aux = 0 #unidades L_sun
  Lg_aux = 0
  Li_aux = 0
  if (in_dist_e[q] > 0):
    for item in in_dist_e_idx[q]:
      Lr_aux = Lr_aux + Lr[item] #unidades L_sun
      Lg_aux = Lg_aux + Lg[item]
      Li_aux = Li_aux + Li[item]

    Lr_dist_e.append(Lr_aux/1e10)
    Lg_dist_e.append(Lg_aux/1e10)
    Li_dist_e.append(Li_aux/1e10)
  else:
    Lr_dist_e.append(999999)
    Lg_dist_e.append(999999)
    Li_dist_e.append(999999)


#magnitude e cor do filamento *******************************************
Mr_dist_e=[]
Mg_dist_e=[]
Mi_dist_e=[]
Mr_dist_e_er=[]
Mg_dist_e_er=[]
Mi_dist_e_er=[]
ptos_dist_e=[]
gr_dist_e=[]
ri_dist_e=[]
gi_dist_e=[]
gr_dist_e_er=[]
ri_dist_e_er=[]
gi_dist_e_er=[]


for q in range(len(dist_e_from_fil)):
  Mr_aux = [] #unidades L_sun
  Mg_aux = []
  Mi_aux = []
  ptos_aux=[]
  if (in_dist_e[q] > 0):
    for item in in_dist_e_idx[q]:
      Mr_aux.append(Mr_clean[item]) #unidades L_sun
      Mg_aux.append(Mg_clean[item])
      Mi_aux.append(Mi_clean[item])
      ptos_aux.append([ptos_clean[item][0],ptos_clean[item][1]])

    ptos_dist_e.append(ptos_aux)
    Mr_dist_e.append(sum(Mr_aux)/in_dist_e[q])
    Mg_dist_e.append(sum(Mg_aux)/in_dist_e[q])
    Mi_dist_e.append(sum(Mi_aux)/in_dist_e[q])


    gr_dist_e.append(Mg_dist_e[q]-Mr_dist_e[q])
    ri_dist_e.append(Mr_dist_e[q]-Mi_dist_e[q])
    gi_dist_e.append(Mg_dist_e[q]-Mi_dist_e[q])

    Mr_dist_e_er.append(np.std(Mr_aux)/np.sqrt(in_dist_e[q]))
    Mg_dist_e_er.append(np.std(Mg_aux)/np.sqrt(in_dist_e[q]))
    Mi_dist_e_er.append(np.std(Mi_aux)/np.sqrt(in_dist_e[q]))
    gr_dist_e_er.append(np.sqrt( Mg_dist_e_er[q]**2 + Mr_dist_e_er[q]**2 ))
    ri_dist_e_er.append(np.sqrt( Mr_dist_e_er[q]**2 + Mi_dist_e_er[q]**2 ))
    gi_dist_e_er.append(np.sqrt( Mg_dist_e_er[q]**2 + Mi_dist_e_er[q]**2 ))


  else:
    ptos_dist_e.append([999999,999999])
    Mr_dist_e.append(999999)
    Mg_dist_e.append(999999)
    Mi_dist_e.append(999999)
    Mr_dist_e_er.append(999999)
    Mg_dist_e_er.append(999999)
    Mi_dist_e_er.append(999999)
    gr_dist_e.append(999999)
    ri_dist_e.append(999999)
    gi_dist_e.append(999999)
    gr_dist_e_er.append(999999)
    ri_dist_e_er.append(999999)
    gi_dist_e_er.append(999999)


# pras diferencas -----------------------------------------------------------
Mr_dif_dist_e=[]
Mg_dif_dist_e=[]
Mi_dif_dist_e=[]
Mr_dif_dist_e_er=[]
Mg_dif_dist_e_er=[]
Mi_dif_dist_e_er=[]

gr_dif_dist_e=[]
ri_dif_dist_e=[]
gi_dif_dist_e=[]
gr_dif_dist_e_er=[]
ri_dif_dist_e_er=[]
gi_dif_dist_e_er=[]

ptos_dif_dist_e=[]
in_dif_dist_e=[]

for q in range(len(dist_e_from_fil)):
  Mr_aux = [] #unidades L_sun
  Mg_aux = []
  Mi_aux = []
  MrErr_aux=[]
  MiErr_aux=[]
  MgErr_aux=[]
  ptos_aux=[]
  if (in_dist_e[q] > 0):
    if (q == 0):
      Mr_dif_dist_e.append(Mr_dist_e[q])
      Mg_dif_dist_e.append(Mg_dist_e[q])
      Mi_dif_dist_e.append(Mi_dist_e[q])
      Mr_dif_dist_e_er.append(Mr_dist_e_er[q])
      Mg_dif_dist_e_er.append(Mg_dist_e_er[q])
      Mi_dif_dist_e_er.append(Mi_dist_e_er[q])
      gr_dif_dist_e.append(gr_dist_e[q])
      ri_dif_dist_e.append(ri_dist_e[q])
      gi_dif_dist_e.append(gi_dist_e[q])
      gr_dif_dist_e_er.append(gr_dist_e_er[q])
      ri_dif_dist_e_er.append(ri_dist_e_er[q])
      gi_dif_dist_e_er.append(gi_dist_e_er[q])
      ptos_aux.append(ptos_dist_e[q])
      in_dif_dist_e.append(len(r_aux))
      ptos_dif_dist_e.append(ptos_aux)

    else:
      for item in in_dist_e_idx[q]:
        if item not in in_dist_e_idx[q-1]:
          Mr_aux.append(Mr_clean[item]) #unidades L_sun
          Mg_aux.append(Mg_clean[item])
          Mi_aux.append(Mi_clean[item])
          ptos_aux.append([ptos_clean[item][0],ptos_clean[item][1]])

      ptos_dif_dist_e.append(ptos_aux)
      in_dif_dist_e.append(len(Mr_aux))
      Mr_dif_dist_e.append(sum(Mr_aux)/in_dif_dist_e[q])
      Mg_dif_dist_e.append(sum(Mg_aux)/in_dif_dist_e[q])
      Mi_dif_dist_e.append(sum(Mi_aux)/in_dif_dist_e[q])

      gr_dif_dist_e.append(Mg_dif_dist_e[q]-Mr_dif_dist_e[q])
      ri_dif_dist_e.append(Mr_dif_dist_e[q]-Mi_dif_dist_e[q])
      gi_dif_dist_e.append(Mg_dif_dist_e[q]-Mi_dif_dist_e[q])

      Mr_dif_dist_e_er.append(np.std(Mr_aux)/np.sqrt(in_dif_dist_e[q]))
      Mg_dif_dist_e_er.append(np.std(Mg_aux)/np.sqrt(in_dif_dist_e[q]))
      Mi_dif_dist_e_er.append(np.std(Mi_aux)/np.sqrt(in_dif_dist_e[q]))
      gr_dif_dist_e_er.append(np.sqrt( Mg_dif_dist_e_er[q]**2 + Mr_dif_dist_e_er[q]**2 ))
      ri_dif_dist_e_er.append(np.sqrt( Mr_dif_dist_e_er[q]**2 + Mi_dif_dist_e_er[q]**2 ))
      gi_dif_dist_e_er.append(np.sqrt( Mg_dif_dist_e_er[q]**2 + Mi_dif_dist_e_er[q]**2 ))


  else:
    ptos_dif_dist_e.append([999999,999999])
    Mr_dif_dist_e.append(999999)
    Mg_dif_dist_e.append(999999)
    Mi_dif_dist_e.append(999999)
    Mr_dif_dist_e_er.append(999999)
    Mg_dif_dist_e_er.append(999999)
    Mi_dif_dist_e_er.append(999999)
    gr_dif_dist_e.append(999999)
    ri_dif_dist_e.append(999999)
    gi_dif_dist_e.append(999999)
    gr_dif_dist_e_er.append(999999)
    ri_dif_dist_e_er.append(999999)
    gi_dif_dist_e_er.append(999999)

#densidade do campo ------------------------------------------------------------------------
#area em (mpc/h)^2
# area_field = area_ellipse(conv, np.asarray(orig['ra']), np.asarray(orig['dec']))/(h*h)
area_field = area_ellipse(conv, np.asarray(orig['ra']), np.asarray(orig['dec']))
densi_field = np.size(orig['ra'])/area_field #num_gal/area (mpc^2)

#densidade relativa ------------------------------------------------------
densi_rlt_e=[]
for q in range(len(dist_e_from_fil)):
  densi_rlt_e.append(densi_dist_e[q]/densi_field)

densi_dif_rlt_e=[]
for q in (range(len(dist_e_from_fil))):
  densi_dif_rlt_e.append(densi_dif_dist_e[q]/densi_field)
#===============================================
# fatias do filamento
#==============================================
#mesma analise do que a superior
#so que dividindo do filamento em partes
n_slc_e=n_slc
ra_slc_e=np.array_split(ra,n_slc_e)
dec_slc_e=np.array_split(dec,n_slc_e)

compri_slc_e=[]
compri_mpc_slc_e=[]
densi_slc_e=[]
in_idx_slc_e=[]
in_slc_e=[]
densi_slc_e=[]
Lr_slc_e=[]
Lg_slc_e=[]
Li_slc_e=[]
Mr_slc_e=[]
Mg_slc_e=[]
Mi_slc_e=[]
gr_slc_e=[]
ri_slc_e=[]
gi_slc_e=[]
Mr_slc_e_er=[]
Mg_slc_e_er=[]
Mi_slc_e_er=[]
gr_slc_e_er=[]
ri_slc_e_er=[]
gi_slc_e_er=[]
densi_rlt_slc_e=[]
ptos_slc_e=[]
densi_slc_e_fil=[]
densi_rlt_slc_e_fil=[]




#faz as contas para cada fatia
for f in range(n_slc_e):
  ptos_slc_e_fil = np.column_stack((ra_slc_e[f],dec_slc_e[f]))
  #calcula valores importantes----------------------------------------
  #fita subarray
  p1=scp.interpolate.interp1d(ra_slc_e[f],dec_slc_e[f])
  xp = np.linspace(min(ra_slc_e[f]), max(ra_slc_e[f]),100)
  aj = p1(xp)
  
  compri_slc_e.append(arc_length(xp, aj))
  # compri_mpc_slc_e.append(deg2Mpc(conv,compri_slc_e[f])/h)
  compri_mpc_slc_e.append(deg2Mpc(conv,compri_slc_e[f]))
  densi_slc_e_fil.append(np.size(ra_slc_e[f])/compri_mpc_slc_e[f]) #numero de galaxias por mpc
  densi_rlt_slc_e_fil.append(densi_slc_e_fil[f]/densi_field)
  #numero de galaxias mais proximas do que 1Mpc/h ao eixo do filamento-----------
  # M=1/h
  M=1
  d=Mpc2deg(conv,M)
  in_idx_slc_e.append(ptos_within(ptos_clean, ptos_slc_e_fil, d))
  in_slc_e.append(np.size(in_idx_slc_e[f]))
  densi_slc_e.append(in_slc_e[f]/(compri_mpc_slc_e[f]*M))
  densi_rlt_slc_e.append(densi_slc_e[f]/densi_field)


  #Luminosidade, magnitude e cor do filamento -------------------------------------------
  #1 Mpc (soma da luminosidade de todas as galáxias nesses range)
  Lr_aux = 0
  Lg_aux = 0
  Li_aux = 0
  r_aux=[]
  g_aux=[]
  i_aux=[]
  aux_ptos_slc_e=[]

  for item in in_idx_slc_e[f]:
    Lr_aux = Lr_aux + Lr[item] #unidades L_sun
    Lg_aux = Lg_aux + Lg[item]
    Li_aux = Li_aux + Li[item]
    r_aux.append(Mr_clean[item])
    g_aux.append(Mg_clean[item])
    i_aux.append(Mi_clean[item])
    aux_ptos_slc_e.append([ptos_clean[item][0],ptos_clean[item][1]])


  if (np.size(in_idx_slc_e[f]) > 0):
    ptos_slc_e.append(aux_ptos_slc_e)
    Lr_slc_e.append(Lr_aux/1e10)
    Lg_slc_e.append(Lg_aux/1e10)
    Li_slc_e.append(Li_aux/1e10)
    Mr_slc_e.append(sum(r_aux)/in_slc_e[f])
    Mg_slc_e.append(sum(g_aux)/in_slc_e[f])
    Mi_slc_e.append(sum(i_aux)/in_slc_e[f])
    gr_slc_e.append(Mg_slc_e[f] - Mr_slc_e[f])
    ri_slc_e.append(Mr_slc_e[f] - Mi_slc_e[f])
    gi_slc_e.append(Mg_slc_e[f] - Mi_slc_e[f])

    Mr_slc_e_er.append(np.std(r_aux)/np.sqrt(in_slc_e[f]))
    Mg_slc_e_er.append(np.std(g_aux)/np.sqrt(in_slc_e[f]))
    Mi_slc_e_er.append(np.std(i_aux)/np.sqrt(in_slc_e[f]))
    gr_slc_e_er.append(np.sqrt( Mg_slc_e_er[f]**2 + Mr_slc_e_er[f]**2 ))
    ri_slc_e_er.append(np.sqrt( Mr_slc_e_er[f]**2 + Mi_slc_e_er[f]**2 ))
    gi_slc_e_er.append(np.sqrt( Mg_slc_e_er[f]**2 + Mi_slc_e_er[f]**2 ))

  else:
    ptos_slc_e.append([999999,999999])
    Mr_slc_e.append(999999)
    Mg_slc_e.append(999999)
    Mi_slc_e.append(999999)
    gr_slc_e.append(999999)
    ri_slc_e.append(999999)
    gi_slc_e.append(999999)
    Lr_slc_e.append(999999)
    Lg_slc_e.append(999999)
    Li_slc_e.append(999999)
    Mr_slc_e_er.append(999999)
    Mg_slc_e_er.append(999999)
    Mi_slc_e_er.append(999999)
    gr_slc_e_er.append(999999)
    ri_slc_e_er.append(999999)
    gi_slc_e_er.append(999999)


# #===============================================
# # salva
# #==============================================
out_name_f2=out_name+'excl_grad.csv'
col0=['0-0.25', '0.25-0.5','0.5-0.75','0.75-1.', '1.-1.25', '1.25-1.5']
salva=pd.DataFrame(columns=['Distance_from _fil (Mpc/h)','Ng', 'Relative Density', 'Lum_r',
      'Lum_g', 'Lum_i', 'Mr', 'Mr_er','Mg', 'Mg_er', 'Mi', 'Mi_er',
      'gr', 'gr_er', 'ri', 'ri_er', 'gi', 'gi_er'])
for q in range(len(densi_dist)):
  salva.loc[q] = dist_e_from_fil[q], in_dist_e[q], densi_rlt[q], Lr_dist_e[q], Lg_dist_e[q], Li_dist_e[q], Mr_dist_e[q], Mr_dist_e_er[q], Mg_dist_e[q], Mg_dist_e_er[q],  Mi_dist_e[q], Mi_dist_e_er[q], gr_dist_e[q], gr_dist_e_er[q], ri_dist_e[q], ri_dist_e_er[q], gi_dist_e[q], gi_dist_e_er[q]
for f in range(len(densi_dif_dist_e)):
  salva.loc[q+f] = col0[f], in_dif_dist_e[f], densi_dif_rlt[f], 0, 0, 0, Mr_dif_dist_e[f], Mr_dif_dist_e_er[f], Mg_dif_dist_e[f], Mg_dif_dist_e_er[f], Mi_dif_dist_e[f], Mi_dif_dist_e_er[f], gr_dif_dist_e[f], gr_dif_dist_e_er[f], ri_dif_dist_e[f], ri_dif_dist_e_er[f], gi_dif_dist_e[f], gi_dif_dist_e_er[f]
salva.to_csv(out_name_f2)

# SALVA GALAXIAS IDENTIFICADAS COMO MEMBRO
out_name_f3=out_name+'excl_grad_membergalaxies.csv'
colra=[]
coldec=[]
for q in range(len(densi_rlt_e)):
  if (q==0):
    aux='ra_0_%s' % dist_e_from_fil[q]
    colra.append(aux)
    aux='dec_0_%s' % dist_e_from_fil[q]
    coldec.append(aux)
  else:
    aux='ra_%s_%s' % (dist_e_from_fil[q-1], dist_e_from_fil[q])
    colra.append(aux)
    aux='dec_%s_%s' % (dist_e_from_fil[q-1], dist_e_from_fil[q])
    coldec.append(aux)

aux_ptos_dist_e=np.array(ptos_dist_e[0]).T
dt={colra[0]: aux_ptos_dist_e[0],coldec[0]: aux_ptos_dist_e[1]}
mbrs=pd.DataFrame(dt,columns=[colra[0],coldec[0]] )

for q in range(len(densi_rlt_e)-1):
  aux_ptos_dist_e=np.array(ptos_dist_e[q+1]).T
  aux_ptos_ants=np.array(ptos_dist_e[q]).T
  auxra=[]
  for item in aux_ptos_dist_e[0]:
    if item not in aux_ptos_ants[0]:
      auxra.append(item)
  auxdec=[]
  for item in aux_ptos_dist_e[1]:
    if item not in aux_ptos_ants[1]:
      auxdec.append(item)
  mbrs[colra[q+1]]=pd.Series(auxra)
  mbrs[coldec[q+1]]=pd.Series(auxdec)

mbrs.to_csv(out_name_f3)

# slices
out_name_f4=out_name+'excl_slices.csv'

col0=[]
for q in range(len(compri_slc_e)):
  col0.append(q+1)

salva=pd.DataFrame(columns=['Group', 'Length (deg)', 'Length (Mpc)','Ng','Relative Density', 'Lum_r', 'Lum_g', 'Lum_i', 'Mr', 'Mr_er','Mg', 'Mg_er', 'Mi', 'Mi_er', 'gr', 'gr_er', 'ri', 'ri_er', 'gr', 'gr_er'])
for q in range(len(densi_slc_e)):
  salva.loc[q] = col0[q], compri_slc_e[q], compri_mpc_slc_e[q], in_slc_e[q], densi_rlt_slc_e[q], Lr_slc_e[q], Lg_slc_e[q], Li_slc_e[q], Mr_slc_e[q], Mr_slc_e_er[q], Mg_slc_e[q], Mg_slc_e_er[q], Mi_slc_e[q], Mi_slc_e_er[q], gr_slc_e[q], gr_slc_e_er[q], ri_slc_e[q], ri_slc_e_er[q], gi_slc_e[q], gi_slc_e_er[q]
salva.to_csv(out_name_f4)



out_name_f5=out_name+'excl_slice_membergalaxies.csv'
colra=[]
coldec=[]
for q in range(len(compri_slc_e)):
  aux='ra_'+'%s' % q
  colra.append(aux)
  aux='dec_'+'%s' % q
  coldec.append(aux)

aux_ptos_slc_e=np.array(ptos_slc_e[0]).T
dt={colra[0]: aux_ptos_dist[0],coldec[0]: aux_ptos_dist[1]}
mbrs=pd.DataFrame(dt,columns=[colra[0],coldec[0]] )

for q in range(len(compri_slc_e)-1):
  aux_ptos=np.array(ptos_slc_e[q+1]).T
  # aux_ptos_ants=np.array(ptos_slc_e[q]).T
  auxra=[]
  print(aux_ptos[0])
  if len(aux_ptos) >2: 
    for item in aux_ptos[0]:
      # if item not in aux_ptos_ants[0]:
      auxra.append(item)
  auxdec=[]
  if len(aux_ptos) >2:
    for item in aux_ptos[1]:
      # if item not in aux_ptos_ants[1]:
      auxdec.append(item)
  mbrs[colra[q+1]]=pd.Series(auxra)
  mbrs[coldec[q+1]]=pd.Series(auxdec)

mbrs.to_csv(out_name_f5)


# #====================================================================
# # PLOTA
# #====================================================================
# #Acha maximos e minimos pras normalizacoes das cores

# max_slc=max(max(i for i in gr_slc if i < 999), max(i for i in gr_slc_e if i < 999))
# min_slc=min(min(gr_slc), min(gr_slc_e))

# max_grad=max(max(i for i in gr_dif_dist if i < 999), max(i for i in gr_dif_dist_e if i < 999))
# min_grad=min(min(gr_dif_dist), min(gr_dif_dist_e))

# # #===============================================
# # # SEM excluir galaxias dos aglomerados
# # #==============================================
# # #-- plot fatias do filamento ----------
# jet = plt.get_cmap('jet')
# cNorm  = colors.Normalize(vmin=min_slc, vmax=max_slc)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
# cores=scalarMap.to_rgba(gr_slc)

# sz=[20,24]*int(n_slc/2)

# markers=['v','^']*int(n_slc/2)


# crit = skm.stdDistortion
# proj = skm.EqualEarth.optimize(ra, dec, crit=crit)
# map = skm.Map(proj)
# sep=0.5
# map.grid(sep=sep)
# len = 10
# size = 100*np.random.rand(len)
# aux_orig=np.array(ptos_orig).T
# map.scatter(aux_orig[0], aux_orig[1],s=3, edgecolor='k', facecolor='k', alpha=0.2)

# for f in range(n_slc):
#   pp=f+1
#   map.scatter(ra_slc[f], dec_slc[f], c=cores[f], marker='.', s=6)
#   aux_slc=np.array(ptos_slc[f]).T
#   map.scatter(aux_slc[0], aux_slc[1], marker=markers[f], c=cores[f], alpha=0.7, s=sz[f], edgecolors=cores[f], label='part %.f' % pp)

# map.scatter(ra_cls, dec_cls, s=160, edgecolor='k', facecolor='k', marker='*')

# cont=input("zoom automatico? [y|n]: ")
# if (cont=='y'): 
#   map.focus(ra, dec)
# else:
#   input("Prosseguir?")
# # map.title('Density with random scatter')
# cb = map.colorbar(scalarMap, cb_label="$g-r$")

# print("qual o deslocamento da escala?")
# des_x=float(input("horizontal: "))
# des_y=float(input("vertical: "))

# #escala
# scl_deg=Mpc2deg(conv,5)
# ra_scl=np.linspace(max(ra)-des_x-scl_deg/2,max(ra)-des_x+scl_deg/2,100)
# dec_scl=np.array([min(dec)+des_y]*100)
# map.plot(ra_scl,dec_scl, color='k')

# input("Pronto?")


# outp1=out_name+'_slices.png'
# map.savefig(outp1)

# # # gradiente --------------------------------------

# jet = plt.get_cmap('jet')
# cNorm  = colors.Normalize(vmin=min_grad, vmax=max_grad)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
# cores=scalarMap.to_rgba(gr_dif_dist)

# crit = skm.stdDistortion
# proj = skm.EqualEarth.optimize(ra, dec, crit=crit)
# map = skm.Map(proj)
# sep=0.5
# map.grid(sep=sep)
# len = 10
# size = 100*np.random.rand(len)
# aux_orig=np.array(ptos_orig).T
# map.scatter(aux_orig[0], aux_orig[1],s=3, edgecolor='k', facecolor='k', alpha=0.2)
# map.scatter(ra, dec, s=6, edgecolor='gray', facecolor='gray', alpha=0.2)


# for f in range(np.size(in_dist)):
#   pp=f+1
#   aux_ptos=np.array(ptos_dif_dist[f]).T
#   map.scatter(aux_ptos[0], aux_ptos[1], c=cores[f], marker='o', s=sz[f], alpha=0.7)
# # map.legend()

# map.scatter(ra_cls, dec_cls, s=120, edgecolor='k', facecolor='k', marker='*')

# cont=input("zoom automatico? [y|n]: ")
# if (cont=='y'): 
#   map.focus(ra, dec)
# else:
#   input("Prosseguir?")

# # map.title('Density with random scatter')
# cb = map.colorbar(scalarMap, cb_label="$g-r$")

# #escala
# scl_deg=Mpc2deg(conv,5)
# ra_scl=np.linspace(max(ra)-des_x-scl_deg/2,max(ra)-des_x+scl_deg/2,100)
# dec_scl=np.array([min(dec)+des_y]*100)
# map.plot(ra_scl,dec_scl, color='k')

# input("Pronto?")


# outp2=out_name+'_grad.png'
# map.savefig(outp2)


# # # #===============================================
# # # # EXCLUINDO galaxias dos aglomerados
# # # #==============================================

# #-- plot fatias do filamento ----------
# jet = plt.get_cmap('jet')
# cNorm  = colors.Normalize(vmin=min_slc, vmax=max_slc)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
# cores=scalarMap.to_rgba(gr_slc_e)
# sz=[20,24]*int(n_slc_e/2)

# markers=['v','^']*int(n_slc_e/2)


# crit = skm.stdDistortion
# proj = skm.EqualEarth.optimize(ra, dec, crit=crit)
# map = skm.Map(proj)
# sep=0.5
# map.grid(sep=sep)
# len = 10
# size = 100*np.random.rand(len)
# # map.scatter(ra, dec, s=8, edgecolor='k', facecolor='k', alpha=0.5)
# aux_clean=np.array(ptos_clean).T
# map.scatter(aux_clean[0], aux_clean[1],s=3, edgecolor='k', facecolor='k', alpha=0.2)
# # aux_excl=ptos_excl.T
# # map.scatter(aux_excl[0], aux_excl[0], s=100, edgecolor='k', facecolor='k', marker='*')

# for f in range(n_slc_e):
#   pp=f+1
#   map.scatter(ra_slc_e[f], dec_slc_e[f], c=cores[f], marker='.', s=6)
#   aux_slc=np.array(ptos_slc_e[f]).T
#   map.scatter(aux_slc[0], aux_slc[1], marker=markers[f], c=cores[f], alpha=0.7, s=sz[f], edgecolors=cores[f], label='part %.f' % pp)
# # map.legend()
# map.scatter(ra_cls, dec_cls, s=120, edgecolor='k', facecolor='k', marker='*')

# # map.title('Density with random scatter')

# cont=input("zoom automatico? [y|n]: ")
# if (cont=='y'): 
#   map.focus(ra, dec)
# else:
#   input("Prosseguir?")

# cb = map.colorbar(scalarMap, cb_label="$g-r$")


# #escala
# scl_deg=Mpc2deg(conv,5)
# ra_scl=np.linspace(max(ra)-des_x-scl_deg/2,max(ra)-des_x+scl_deg/2,100)
# dec_scl=np.array([min(dec)+des_y]*100)
# map.plot(ra_scl,dec_scl, color='k')

# input("Pronto?")


# outp1=out_name+'excl_slices.png'
# map.savefig(outp1)

# # PLOTA GRADIENTE --------------------------------------
# # legenda=[]

# jet = plt.get_cmap('jet')
# cNorm  = colors.Normalize(vmin=min_grad, vmax=max_grad)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
# cores=scalarMap.to_rgba(gr_dif_dist_e)

# crit = skm.stdDistortion
# proj = skm.EqualEarth.optimize(ra, dec, crit=crit)
# map = skm.Map(proj)
# sep=0.5
# map.grid(sep=sep)
# len = 10
# size = 100*np.random.rand(len)
# aux_clean=np.array(ptos_clean).T
# map.scatter(aux_clean[0], aux_clean[1],s=3, edgecolor='k', facecolor='k', alpha=0.2)
# map.scatter(ra, dec, s=6, edgecolor='gray', facecolor='gray', alpha=0.2)


# for f in range(np.size(in_dist_e)):
#   pp=f+1
#   aux_ptos=np.array(ptos_dif_dist_e[f]).T
#   map.scatter(aux_ptos[0], aux_ptos[1], c=cores[f], marker='o', s=sz[f], alpha=0.7)
# # map.legend()

# map.scatter(ra_cls, dec_cls, s=120, edgecolor='k', facecolor='k', marker='*')

# cont=input("zoom automatico? [y|n]: ")
# if (cont=='y'): 
#   map.focus(ra, dec)
# else:
#   input("Prosseguir?")


# # map.title('Density with random scatter')
# cb = map.colorbar(scalarMap, cb_label="$g-r$")

# #escala
# scl_deg=Mpc2deg(conv,5)
# ra_scl=np.linspace(max(ra)-des_x-scl_deg/2,max(ra)-des_x+scl_deg/2,100)
# dec_scl=np.array([min(dec)+des_y]*100)
# map.plot(ra_scl,dec_scl, color='k')

# input("Pronto?")

# outp2=out_name+'excl_grad.png'
# map.savefig(outp2)


# ============================================================
# ANALISE CDFs
# ===========================================================


#usando vale so do filamento:
gi_total=np.array(Mg) - np.array(Mi)
n_cl=0
dist_galcl_0=[]
dist_galcl_min=[]
lni=np.size(idx_total_e)
print(lni)
for d in range(lni):
  dist_galcl_0.append(dist_galcl_mpc_e[d][n_cl])
  dist_galcl_min.append(np.min(dist_galcl_mpc_e[d]))


print('')
print('Análise de cor das galáxias')
print('com limite entre azuis e vermelhas')
print('dada pelas apenas pelas galáxias do filamento')
anal_cor_fil(gi_gals_e ,Mg_gals_e, Mi_gals_e, gi_total, dist_galfil_mpc_e, dist_galcl_min, 'gi_fil', 'FIL')

#usando vale do campo:
print('')
print('Análise de cor das galáxias')
print('com limite entre azuis e vermelhas')
print('dada por todas as galáxias do campo')
green_val=anal_cor_fil(gi_gals_e ,Mg_gals_e, Mi_gals_e, gi_total, dist_galfil_mpc_e, dist_galcl_min, 'gi_campo', 'FIELD')



#============================================================
# ANALISE GALAXIAS DO AGLOMERADO CENTRAL+ pts excluidos do filamento
#===========================================================
#galaxias do aglomerado central
ptos_agl=[]
Mr_agl=[]
Mg_agl=[]
Mi_agl=[]
z_agl=[]
zErr_agl=[]
agl_ref=[]
dist_Gal2Cl=[]
dist_Gal2Fil=[]
dist_GalFil2Cl=[]
for n in range(len(r_excl)):
  for q in range(len(ptos_orig)):
    if q in idx_agl[n]:
      agl_ref.append(n+1)
      ptos_agl.append([ptos_orig[q][0],ptos_orig[q][1]])
      Mr_agl.append(Mr[q])
      Mg_agl.append(Mg[q])
      Mi_agl.append(Mi[q])
      z_agl.append(z[q])
      zErr_agl.append(zErr[q])
      aux_dist=np.where(np.array(idx_agl[n])==q)[0][0]
      dist_Gal2Cl.append(dist_agl[n][aux_dist])
      dist_Gal2Fil.append(dist_filagl[n][aux_dist])
      dist_GalFil2Cl.append(dist_galfil_agl[n][aux_dist]) 


ra_agl=(np.array(ptos_agl).T)[0]
dec_agl=(np.array(ptos_agl).T)[1]


gr_agl=(np.array(Mg_agl) - np.array(Mr_agl))
# gr_agl_er.append(np.sqrt( MgErr_agl[q]**2 + MrErr_agl[q]**2 ))
gi_agl=(np.array(Mg_agl) - np.array(Mi_agl))
# gi_agl_er.append(np.sqrt( MgErr_agl[q]**2 + MiErr_agl[q]**2 ))
ri_agl=(np.array(Mr_agl) - np.array(Mi_agl))
# ri_agl_er.append(np.sqrt( MrErr_agl[q]**2 + MiErr_agl[q]**2 ))

#Para caa aglomerado, conta quantas galaxias tem em r_excl/2 e na parte restante
Ng_r05=[]
BF_r05=[]
BFerr_r05=[]
gi_r05_avrg=[]
gi_r05_std=[]
Ng_rtot=[]
BF_rtot=[]
BFerr_rtot=[]
gi_rtot_avrg=[]
gi_rtot_std=[]
R_out=[]

for n in range(len(r_excl)):
  d=deg2Mpc(conv,r_excl[n])
  Ng_r05_aux=0
  Ng_r1_aux=0
  cor_r05_aux=[]
  cor_r1_aux=[]
  R_out_aux=[]
  for q in range(len(ptos_agl)):
    if agl_ref[q] == n+1:
      if dist_Gal2Cl[q] <=(d/2):
        Ng_r05_aux +=1
        cor_r05_aux.append(gi_agl[q])
      else:
        Ng_r1_aux +=1
        cor_r1_aux.append(gi_agl[q])
        R_out_aux.append(dist_Gal2Cl[q])

  blue_r05_aux=0
  for f in cor_r05_aux:
    if f < green_val:
      blue_r05_aux+=1
  if Ng_r05_aux == 0 or blue_r05_aux==0:
    BF_r05_aux=0
    BFerr_r05_aux=0
  else:
    BF_r05_aux=blue_r05_aux/Ng_r05_aux
    BFerr_r05_aux=BF_r05_aux*np.sqrt(((1/blue_r05_aux) + (1/Ng_r05_aux)))

  blue_r1_aux=0
  for f in cor_r1_aux:
    if f < green_val:
      blue_r1_aux+=1
  if Ng_r1_aux ==0 or blue_r1_aux==0:
    BF_r1_aux=0
    BFerr_r1_aux=0
  else:
    BF_r1_aux=blue_r1_aux/Ng_r1_aux
    BFerr_r1_aux=BF_r1_aux*np.sqrt(((1/blue_r1_aux) + (1/Ng_r1_aux)))

  R_out.append(max(R_out_aux))
  Ng_r05.append(Ng_r05_aux)
  BF_r05.append(BF_r05_aux)
  BFerr_r05.append(BFerr_r05_aux)
  gi_r05_avrg.append(np.mean(cor_r05_aux))
  gi_r05_std.append(np.std(cor_r05_aux))
  Ng_rtot.append(Ng_r1_aux)
  BF_rtot.append(BF_r1_aux)
  BFerr_rtot.append(BFerr_r1_aux)
  gi_rtot_avrg.append(np.mean(cor_r1_aux))
  gi_rtot_std.append(np.std(cor_r1_aux))

agl_ref2=np.arange(1, len(r_excl)+1, 1, int)
R_in=r_excl/2



out_name_agl=out_name+'_GalsInClus.csv'
col0=['Agl_Ref','RA', 'DEC','z','zErr', 'Mr', 'Mi', 'Mg', 'gr', 'gi', 'ri', 'Dist2Cluster (Mpc)', 'Dist2Fil (Mpc)','DistFil2Cluster (Mpc)']

cols=[agl_ref,ra_agl, dec_agl, z_agl, zErr_agl, Mr_agl, Mg_agl, Mi_agl, gr_agl, gi_agl, ri_agl, dist_Gal2Cl, dist_Gal2Fil, dist_GalFil2Cl]
list_df = pd.DataFrame([ pd.Series(value) for value in cols ]).transpose()

list_df.to_csv(out_name_agl, sep=',', header=col0, index=False)




out_name_agl=out_name+'_GalsInClus_BF.csv'
col0=['Agl_Ref','RA', 'DEC','z', 'R_in (Mpc)', 'Ng_in', 'BF_in', 'BFerr_in', 'gi_avrg_in', 'gi_std_in', 'R_out (Mpc)', 'Ng_out', 'BF_out', 'BFerr_out', 'gi_avrg_out', 'gi_avrg_out']

cols=[agl_ref2,ra_cls, dec_cls, z_excl, R_in, Ng_r05, BF_r05, BFerr_r05, gi_r05_avrg, gi_r05_std, R_out, Ng_rtot, BF_rtot, BFerr_rtot, gi_rtot_avrg, gi_rtot_std]
list_df = pd.DataFrame([ pd.Series(value) for value in cols ]).transpose()

list_df.to_csv(out_name_agl, sep=',', header=col0, index=False)


#===============================================
#SALVA GALAXIAS MENBRO FILAMENTO
#===============================================
distF_Gal2Cl, distF_Gal2Fil, distF_GalFil2Cl=GalsNotinCl_quadrado(idx_total_e,ptos_excl, ptos_fil,ptos_clean,conv)


ra_fil=(np.array(ptos_clean_fil).T)[0]
dec_fil=(np.array(ptos_clean_fil).T)[1]

# idx_galfil=[]
# for n in range(len(ra_fil)):
#   idx_galfil.append(closest(ra_fil[n], dec_fil[n], ptos_fil))

# dist_Gal2Fil=[]
# for n in range(len(ra_fil)):
#   if n in idx_galfil:
#     dist_Gal2Fil.append(distancia_Mpc_1pto(ptos_clean_fil[n], ptos_fil[idx_galfil[n]], conv)) #em Mpc



out_name_fil=out_name+'_GalsInFil.csv'
col0=['RA', 'DEC','z','zErr', 'Mr', 'Mi', 'Mg', 'gr', 'gi', 'ri', 'Dist2Cluster (Mpc)', 'Dist2Fil (Mpc)','DistFil2Cluster (Mpc)']
cols=[ra_fil, dec_fil, z_gals_e, zErr_gals_e, Mg_gals_e, Mi_gals_e, Mr_gals_e, gr_gals_e, gi_gals_e, ri_gals_e, distF_Gal2Cl, distF_Gal2Fil, distF_GalFil2Cl]

list_df = pd.DataFrame([ pd.Series(value) for value in cols ]).transpose()
list_df.to_csv(out_name_fil, sep=',', header=col0, index=False)

