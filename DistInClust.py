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

def closest(ptora, ptodec, array):
  dist=[]
  for item in array:
    d=np.sqrt((ptora-item[0])**2 + (ptodec - item[1])**2)
    dist.append(d)
  d_aux=np.sort(dist)
  menor=d_aux[0]
  arg=dist.index(menor)

  return(arg)

def closest_10(ptora, ptodec, array):
  dist=[]
  for item in array:
    d=np.sqrt((ptora-item[0])**2 + (ptodec - item[1])**2)
    dist.append(d)
  d_aux=np.sort(dist)
  menor10=d_aux[1:21]
  arg10=[]
  for n in range(len(menor10)):
    arg10.append(dist.index(menor10[n]))
  return arg10

  return(arg)

def distancia_Mpc_1pto(pto1, pto2, conv):
  d=np.sqrt((pto1[0]-pto2[0])**2 + (pto1[1] - pto2[1])**2)
  d_Mpc=d*conv*3600/1000

  return d_Mpc

def distancia_1pto(pto1, pto2):
  d=np.sqrt((pto1[0]-pto2[0])**2 + (pto1[1] - pto2[1])**2)
  return d

def divide_lados(x0,y0,ptos_fil):
#acha ponto em fil mais proximo de (x0,y0)
  idx0=closest(x0, y0, ptos_fil)
  p0=ptos_fil[idx0]

#Acha 10 pontos do filamento mais proximo de (idx0)
  idx_closest=closest_10(ptos_fil[idx0][0], ptos_fil[idx0][1], ptos_fil)


#tenta dividir em direita e esquerda
# pra isso, idx0 e idx1 tem que satisfazer 1 AND 2 AND (3 OR/AND 4):
# 1) idx1<idx0<idx2 ou idx2<idx0<idx1
# 2) dist(idx0,idx1)>dist([x0,y0],idx0) and dist(idx0,idx1)>dist([x0,y0],idx1)
# 3) x0-Xidx0 tem sinal diferente de x0-Xidx1
# 4) y0-Yidx0 tem sinal diferente de y0-Yidx1
# se nao satisfizer, é pq os dois pontos são do mesmo lado.
  # busca outros dois ptos mais proximos de (x0,y0)

#loop sobre os 10 indices mais proximos
#vai olhando de 2 em 2
  satisfez=0 #não
  max_inter=np.int(len(idx_closest)-2)
  n0=0
  n=0
  while (n <=max_inter) and (satisfez==0) :
    print('')
    print(n)
    p1=ptos_fil[idx_closest[n0]]
    p2=ptos_fil[idx_closest[n+1]]
    idx1=idx_closest[n0]
    idx2=idx_closest[n+1]

    cond1=0
    if ((idx_closest[n] < idx0) and (idx0 < idx_closest[n+1])) or ((idx_closest[n] > idx0) and (idx0 > idx_closest[n+1])):
      cond1=1


    d_x0x1=distancia_1pto(p0,p1)
    d_x0x2=distancia_1pto(p0,p2)
    d_x1x2=distancia_1pto(p1,p2)

    cond2=0 #nao
    if (d_x1x2 >= d_x0x1) and (d_x1x2 >= d_x0x1):
      cond2=1

    cond3=0
    s_x0x1=(x0 - ptos_fil[idx1][0])/np.abs((x0 - ptos_fil[idx1][0]))
    s_x0x2=(x0 - ptos_fil[idx2][0])/np.abs((x0 - ptos_fil[idx2][0]))
    if np.sign(s_x0x1) != np.sign(s_x0x2):
      cond3=1

    cond4=0
    s_y0y1=(y0 - ptos_fil[idx1][1])/np.abs((y0 - ptos_fil[idx1][1]))
    s_y0y2=(y0 - ptos_fil[idx2][1])/np.abs((y0 - ptos_fil[idx2][1]))
    if np.sign(s_y0y1) != np.sign(s_y0y2):
      cond4=1

    conds=cond1+cond2+cond3+cond4
    if conds >= 2:
    # if (cond1 == 1) and (cond2 == 1) and (cond3 == 1 or cond4 ==1):
    # if (cond1 == 1) and (cond2 == 1):

      satisfez=1
      idx_certo=n
      n = max_inter+1
    else:
      n +=1

  if satisfez ==0:
    idx_certo=0
  idx1=idx_closest[idx_certo]
  idx2=idx_closest[idx_certo+1]


  return satisfez, idx0, idx1, idx2


def distancia_quadrado(dist_deg, x0,y0,ptos_fil, conv):
#devolve indice de dos pontos do filamento numa distancia 
# menor do que dist_mpc de (x0,y0)
  #divide filamento em dois lados, um pra cado de x0, y0.
  #tem que fazer um loop pra verificar
  out = divide_lados(x0,y0,ptos_fil)
  lados2=out[0]
  idx_cl=out[1]
  idx1=out[2]
  idx2=out[3]
  pcl=ptos_fil[idx_cl]

  if lados2 ==1:
    print('')
    print('----------------------')
    print('Aglomerado está no meio de um filamento')
    print('ra = %.2f, dec= %.2f' %(x0,y0))
    print('')
    print('----------------------')
  else:
    print('')
    print('----------------------')
    print('Aglomerado está na ponta de um filamento')
    print('ra = %.2f, dec= %.2f' %(x0,y0))
    print('')
    print('----------------------')

  #conta a distância filamentar dos pontos de cada lado
  # para qnd atingir dist_mpc
  if lados2 == 1:
    idx_excl=[]
    dist_filp0=[]
    idx_excl.append(idx_cl)
    dist_filp0.append(0)
    n0=0
    #lado 1
    if (idx1 < idx_cl):
      dist_lado1=0
      n0=idx_cl
      while (dist_lado1 < dist_deg) and (n0 <=len(ptos_fil)): 
        dist_aux=distancia_1pto(ptos_fil[n0], ptos_fil[n0-1])
        dist_lado1 = dist_lado1 + dist_aux
        if (dist_lado1 < dist_deg):
          n0 -= 1
          idx_excl.append(n0)
          dist_filp0.append(dist_lado1)
        else:
          dist_lado1=999999
      n0=idx_cl
      dist_lado2=0
      while (dist_lado2 < dist_deg) and (n0 >=0): 
        dist_aux=distancia_1pto(ptos_fil[n0], ptos_fil[n0+1])
        dist_lado2 = dist_lado2 + dist_aux
        if (dist_lado2 < dist_deg):
          n0 += 1
          idx_excl.append(n0)
          dist_filp0.append(dist_lado2)
        else:
          dist_lado2=999999

    #lado 2
    else:
      dist_lado1=0
      n0=idx_cl
      while (dist_lado1 < dist_deg) and (n0 >=0): 
        dist_aux=distancia_1pto(ptos_fil[n0], ptos_fil[n0+1])
        dist_lado1 = dist_lado1 + dist_aux
        if (dist_lado1 < dist_deg):
          n0 -= 1
          print(n0)
          idx_excl.append(n0)
          dist_filp0.append(dist_lado1)
        else:
          dist_lado1=999999
      n0=idx_cl
      dist_lado2=0
      while (dist_lado2 < dist_deg) and (n0 <=len(ptos_fil)): 
        dist_aux=distancia_1pto(ptos_fil[n0], ptos_fil[n0-1])
        dist_lado2 = dist_lado2 + dist_aux
        if (dist_lado2 < dist_deg):
          n0 += 1
          idx_excl.append(n0)
          dist_filp0.append(dist_lado2)
        else:
          dist_lado2=999999

  else:
    idx_excl=[]
    idx_excl.append(idx_cl)
    n0=idx_cl
    dist_lado=0
    dist_filp0=[]
    if idx1 < idx_cl:
      while (dist_lado < dist_deg) and (n0 <=len(ptos_fil)): 
        dist_aux=distancia_1pto(ptos_fil[n0], ptos_fil[n0-1])
        dist_lado = dist_lado + dist_aux
        if (dist_lado < dist_deg):
          n0 -= 1
          idx_excl.append(n0)
          dist_filp0.append(dist_lado)
        else:
          dist_lado=999999
    else:
      while (dist_lado < dist_deg) and (n0 >=0): 
        dist_aux=distancia_1pto(ptos_fil[n0], ptos_fil[n0+1])
        dist_lado = dist_lado + dist_aux
        if (dist_lado < dist_deg):
          n0 += 1
          idx_excl.append(n0)
          dist_filp0.append(dist_lado)
        else:
          dist_lado=999999


  #retorna indices do filamento que estao mais proximos do que dist_mpc
  return idx_excl,dist_filp0


def GalsinCl_quadrado(dist_deg, x0, y0, ptos_fil,ptos_gal,conv):

  #As galaxias do aglomerado são idx_galsinR UNION (idx_galsinSqr INTERSECTION idx_in15)

  # 1)-------------------------------------------------------
  #Acha index das galaxias dentro de um raio= dist_deg do centro do aglomerado
  n=0
  idx_galsinR=[]
  for pg in ptos_gal:
    aux_dist=distancia_1pto([x0,y0], pg)
    if aux_dist <dist_deg:
      idx_galsinR.append(n)
    n +=1

  # 2)-------------------------------------------------------
  #Acha index das galaxias que estão mais em em d<1.5Mpc do filamento
  d=Mpc2deg(conv,1.5)
  idx_in15 = ptos_within(ptos_gal, ptos_fil, d)

  # 3) -------------------------------------------------------
  #Acha index dos pontos do filamento em dist < dist_deg
  idx_FilInSq,dist_FilInSq=distancia_quadrado(dist_deg, x0,y0,ptos_fil, conv)

  print('----------------')
  print('indice dos pontos do filamento perto do agl')
  print(idx_FilInSq)
  print('----------------')

  #para todas as galaxias, acha index do pto do filamento mais proximo
  idx_galfil=[]
  for item in ptos_gal:
    idx_galfil.append(closest(item[0], item[1], ptos_fil))

  print('----------------')
  print('indice dos pontos do filamento mais proximos das galaxias')
  print(idx_galfil)
  print('----------------')

  idx_GalInSq=[]
  for n in range(len(idx_galfil)):
    if idx_galfil[n] in idx_FilInSq:
      idx_GalInSq.append(n)

  # 4) -------------------------------------------------------
  #combina as listas de indices
  intersec = list(filter(lambda x: x in idx_GalInSq, idx_in15))
  idx_GalInCl = list(set().union(intersec,idx_galsinR))
  print('----------------')
  print('indice das galaxias no aglomerado')
  print(idx_GalInCl)
  print('----------------')
  # 5) -------------------------------------------------------
  #acha distancia filamento_agl dos ptos do agl
  dist_GalFil2Cl=[] #meor distancia do pto do limaneto (correspondente à galaxia) até o aglomerado
  for n in idx_GalInCl:
    # if idx_galfil[n] in idx_FilInSq:
    print(n)
    pfil=idx_galfil[n]
    dist_GalFil2Cl.append(distancia_Mpc_1pto(ptos_fil[pfil], [x0,y0], conv))

    # aux_dist=np.where(np.array(idx_FilInSq)==idx_galfil[n])[0][0]
    # dist_GalFil2Cl.append(deg2Mpc(conv,dist_FilInSq[aux_dist])) #Mpc

  # 6) -------------------------------------------------------
  #acha distancia gas galaxias ao filamento e ao agl
  dist_Gal2Cl=[] #menor distancia da galaxia ao aglomerado
  dist_Gal2Fil=[] #menor distancia da galaxia ao filamento
  for n in idx_GalInCl:
    pg=n
    pfil==idx_galfil[n]
    dist_Gal2Cl.append(distancia_Mpc_1pto(ptos_gal[pg], [x0,y0], conv))#em Mpc
    dist_Gal2Fil.append(distancia_Mpc_1pto(ptos_gal[pg], ptos_fil[pfil], conv)) #em Mpc

  # 7) -------------------------------------------------------
  #acha indices  das galaxias que nao estão no aglomerado
  idx_GalNotCl=[]
  for n in range(len(idx_galfil)):
    if n not in idx_GalInCl:
      idx_GalNotCl.append(n)

  return idx_GalNotCl, idx_GalInCl, dist_Gal2Cl, dist_Gal2Fil, dist_GalFil2Cl


def GalsNotinCl_quadrado(idx_GinF, pos_cls, ptos_fil,ptos_gal,conv):

  ptos_gal=np.array(ptos_gal)
  #acha galaxias mais proximas do que 1.5 Mpc do fil
  # d=Mpc2deg(conv,1.5)
  # idx_in15 = ptos_within(ptos_gal, ptos_fil, d)
  #acha index do pto do filamento mais proximo às galaxias in1.5
  idx_galfil=[]
  for n in idx_GinF:
    n=int(n)
    idx_galfil.append(closest(ptos_gal[n][0], ptos_gal[n][1], ptos_fil))

  dist_Gal2Fil=[] 
  for n in range(len(idx_GinF)):
    valg=int(idx_GinF[n])
    valf=idx_galfil[n]
    dist_Gal2Fil.append(distancia_Mpc_1pto(ptos_gal[valg], ptos_fil[valf], conv)) #em Mpc


  dist_GalFil2Cl=[]
  dist_Gal2Cl=[]
  for n in range(len(idx_GinF)):
    valg=int(idx_GinF[n])
    valf=idx_galfil[n]
    aux_GalFil2Cl=[]
    aux_Gal2Cl=[]
    for pcl in pos_cls:
      aux_GalFil2Cl.append(distancia_Mpc_1pto(ptos_fil[valf], pcl, conv))
      aux_Gal2Cl.append(distancia_Mpc_1pto(ptos_gal[valg], pcl, conv))#em Mpc
    dist_Gal2Cl.append(min(aux_Gal2Cl))
    dist_GalFil2Cl.append(min(aux_GalFil2Cl))


  return  dist_Gal2Cl, dist_Gal2Fil, dist_GalFil2Cl