# Identifica quais galáxias pertencem aos aglomerados
#===================================================

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



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FUNÇOES
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def closest(ptora, ptodec, array):
  dist=[]
  for item in array:
    d=np.sqrt((ptora-item[0])**2 + (ptodec - item[1])**2)
    dist.append(d)
  d_aux=np.sort(dist)
  menor=d_aux[0]
  arg=dist.index(menor)

  return(arg)
#****************************

def distancia_Mpc(ptora, ptodec, array, conv):
  dist=[]
  for item in array:
    d=np.sqrt((ptora-item[0])**2 + (ptodec - item[1])**2)
    dist.append(d)
  d_aux=np.sort(dist)
  menor=d_aux[0]
  menor_Mpc=menor*conv*3600/1000

  return menor, menor_Mpc
#****************************

def ptos_within(ptos_orig, ptos_fil, dist):
    # build the KDTree using the *smaller* points array
    tree = spatial.cKDTree(ptos_fil)
    groups = tree.query_ball_point(ptos_orig, dist)
    indices = np.unique([i for i, grp in enumerate(groups) if np.size(grp)])
    return indices
#****************************

def area_ellipse(conv, ra, dec):
  r_ra = max(ra) - min(ra)
  r_dec = max(dec) - min(dec)

  r_ra_mpc = deg2Mpc(conv, r_ra)
  r_dec_mpc = deg2Mpc(conv, r_dec)

  area=np.pi*r_ra_mpc*r_dec_mpc
  return area #mpc^2

#****************************

def deg2Mpc(conv,comp_deg):
  comp_mpc = comp_deg*conv*3600/1000
  return comp_mpc
#****************************

def Mpc2deg(conv,comp_mpc):
  comp_deg=comp_mpc/(conv*3600/1000)
  return comp_deg
#****************************

def dist2Cl(dt_gals,ptos_excl,conv):
  aux_degcl=[]
  aux_mpccl=[]
  ra=dt_gals['ra'].values
  dec=dt_gals['dec'].values
  for d in range(len(dt_gals)):
    xf=ra[d]
    yf=dec[d]
    #distancia ao aglomerado mais proximo
    aux_degc=[]
    aux_mpcc=[]
    for d in range(len(ptos_excl)):
      ptos_cl=np.array([ptos_excl[d]])
      aux_deg, aux_mpc=distancia_Mpc(xf, yf, ptos_cl, conv)
      aux_degc.append(aux_deg)
      aux_mpcc.append(aux_mpc)
    aux_degcl.append(min(aux_degc))
    aux_mpccl.append(min(aux_mpcc))

  return aux_degcl, aux_mpccl
#****************************

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MAIN
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def AnalCl(dt_g,ang,lin, conv):

  excl_cls='real_cls_in_fils.csv'

  excl=pd.read_csv(excl_cls, delimiter=',')
  ptos_excl = excl.as_matrix(columns=excl.columns[1:3])

  #limpa dados usando o ajuste da RS
  # orig=pd.read_csv(in_data_orig, delimiter=',')
  orig=dt_g
  gr_sup=ang*(orig['Mr'].values) + lin +0.15
  dTot=orig.query('(-24.5 < Mr < -18) & (-0.5 < gr < @gr_sup)')


  dClean = dTot.query('member == 1')
  dClean.reset_index(drop=True,inplace=True)
  ptos_clean = dClean.as_matrix(columns=dClean.columns[0:2])


  #TODAS AS GALAXIAS DO FILAMENTO
  degcl,mpccl= dist2Cl(dClean,ptos_excl,conv)
  dClean['Dist2ClosCluster_deg']=degcl
  dClean['Dist2ClosCluster_Mpc']=mpccl


  # salva tudo
  # GALAXIAS GRAD
  dClean.to_csv('GalsInCl_Clean_final.csv')

  return 

