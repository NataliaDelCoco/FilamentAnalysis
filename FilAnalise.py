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
from anal_cor import anal_cor_fil
from DistInClust import GalsinCl_quadrado
from ProjCorr import ProjCorrFil_unico


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
  bg=np.where(ra == max(ra))[0][0]
  sm=np.where(ra == min(ra))[0][0]

  r_ra = arc_length( [ra[sm],ra[bg]], [dec[sm],dec[bg]] )/2
  r_dec = (max(dec) - min(dec))/2

  r_ra_mpc = deg2Mpc(conv, r_ra)
  r_dec_mpc = deg2Mpc(conv, r_dec)

  area=np.pi*r_ra_mpc*r_dec_mpc
  return area #mpc^2
#****************************

def arc_length(ra, dec):
  ra_r=np.radians(ra)
  dec_r=np.radians(dec)
  npts = np.size(ra)
  arc=0
  for i in range(npts-1):
    if ra[i] == ra[i+1]:
      if dec[i] == dec[i+1]:
        s=2
    else:
      s=1
    print('')
    print(i)
    ax1=np.sin(dec_r[i])*np.sin(dec_r[i+s])
    ax2=np.cos(dec_r[i])*np.cos(dec_r[i+s])*np.cos(ra_r[i]-ra_r[i+s])
    arccos=np.arccos(ax1+ax2)
    print(arccos)
    arc=arc+arccos
    print(arc)

  arc_deg=np.degrees(arc)
  # arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
  # for k in range(1, npts):
  #     arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)

  return arc_deg
#****************************

def deg2Mpc(conv,comp_deg):
  comp_mpc = comp_deg*conv*3600/1000
  return comp_mpc
#****************************

def Mpc2deg(conv,comp_mpc):
  comp_deg=comp_mpc/(conv*3600/1000)
  return comp_deg
#****************************

def ordena_fil(in_fint, in_data):
  fint=pd.read_csv(in_fint, delimiter=',',header=None)
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

  return (ra,dec)
#****************************

def index_GalsinCluster(excl_cls,ptos_fil,dTot,conv):

  excl=pd.read_csv(excl_cls, delimiter=',')
  ptos_excl = excl.as_matrix(columns=excl.columns[1:3]) #ra,dec
  r_excl = np.asarray(excl['R(arcmin)'])/60 #GRAUS ptos dentro do raio R serao excluidos
  z_excl = np.asarray(excl['Redshift']) #redshift
  ra_cls = np.asarray(excl['RA'])
  dec_cls = np.asarray(excl['DEC'])

  ptos_orig = np.array([dTot.ra.values, dTot.dec.values]).T

  idx_InCl=[]
  for q in range(len(r_excl)):
    aux_out,aux_agl, aux_dist, aux_dist2, aux_dist3=GalsinCl_quadrado(r_excl[q], ra_cls[q], dec_cls[q], ptos_fil,ptos_orig, conv)
    idx_InCl.append(aux_agl)

  return idx_InCl
#****************************

def fil_properties(ra_fil, dec_fil,conv,area_field, n_field):
  #divide o array inicial em subarrays
  #IGUAL PARA AS DUAS PARTES
  # n_subs=30
  # ra_subs=np.array_split(ra_fil,n_subs)
  # dec_subs=np.array_split(dec_fil,n_subs)


  # #fita cada um dos subarrays (interp)
  # f_subs=[]
  # xp_subs=[]
  # aj_subs=[]
  # for q in range(n_subs):
  #   f_subs.append(scp.interpolate.interp1d(ra_subs[q],dec_subs[q]))
  #   xp_subs.append(np.linspace(min(ra_subs[q]), max(ra_subs[q]),5))
  #   aux_f=f_subs[q]
  #   aj_subs.append(aux_f(xp_subs[q]))

  # #junta tudo
  # xp = list(itertools.chain.from_iterable(xp_subs)) #ra_fil
  # aj = list(itertools.chain.from_iterable(aj_subs)) #dec_fil

  # for q in range(n_subs):
  #   plt.plot(ra_subs[q], dec_subs[q], '.', xp_subs[q], aj_subs[q], '-')
  # plt.show()
  # plt.close()

  #Acha limites superiores e inferiores do comprimento
  #Lim inf = reta entre primeiro e ultimo ponto
  # compri_inf = arc_length( [xp[0],xp[-1]], [aj[0],aj[-1]] )
  compri_inf = arc_length( [ra_fil[0],ra_fil[-1]], [dec_fil[0],dec_fil[-1]] )
  #Lim sup = seimi-circulo de raio compri_inf/2

  compri_sup = 2*np.pi*(compri_inf/2)

  #calcula valores importantes----------------------------------------
  # compri = arc_length(xp, aj)
  compri = arc_length(ra_fil, dec_fil)
  # compri_mpc = deg2Mpc(conv,compri)/h
  compri_mpc = deg2Mpc(conv,compri)
  compri_inf_mpc = deg2Mpc(conv, compri_inf)
  compri_sup_mpc = deg2Mpc(conv, compri_sup) 
  compriErr_inf = np.abs(compri_mpc - compri_inf_mpc)
  compriErr_sup = np.abs(compri_mpc - compri_sup_mpc)

  densi_fil = len(ra_fil)/(compri_mpc*1.5) #numero de galaxias por mpc²
  densi_fil_inf = len(ra_fil)/(compri_sup_mpc*1.5)
  densi_fil_sup = len(ra_fil)/(compri_inf_mpc*1.5)


  #densidade relativa ------------------------------------------------------
  # area_field = area_ellipse(conv, np.asarray(orig['ra']), np.asarray(orig['dec']))/(h*h)
  densi_field=n_field/area_field
  densi_fil_rlt=densi_fil/densi_field
  densi_fil_rlt_inf=densi_fil_inf/densi_field
  densi_fil_rlt_sup=densi_fil_sup/densi_field
  densi_fil_rltErr_inf=np.abs(densi_fil_rlt-densi_fil_rlt_inf)
  densi_fil_rltErr_sup= np.abs(-densi_fil_rlt+densi_fil_rlt_sup)



  # SALVA AS SAIDAS EM FILES

  head1=('Length (deg), Length (Mpc), LengthErrSup, LengthErrInf, FieldDensity, FieldNg, FieldArea' )
  val=[round(compri,4), round(compri_mpc,4), round(compriErr_sup,4), round(compriErr_inf,4), round(densi_field,4),n_field, area_field]
  out_name_f1='f1_comrpimento.txt'
  with open(out_name_f1,'w') as t:
    t.write(head1)
    t.write("\n")
    for y in range(len(val)):
      v=str(val[y])
      t.write(v)
      if (y < (len(val)-1)): 
        t.write(", ")
  t.close()

  return compri_mpc, compri_sup_mpc, compri_inf_mpc
  #****************************

def Gradiente(dt_gals, ptos_fil,compri_mpc,compri_mpcS,compri_mpcI,conv):

  # dt_gals=dt_gals.reset_index(inplace=True)
  ptos_orig=np.array([dt_gals['ra'].values,dt_gals['dec'].values]).T

  # dist_from_fil=[0.25,0.5,0.75,1.,1.25,1.5]
  dist_from_fil=[0.5,1.,1.5]

  Grad_gals=[] #galaxias em fatias cilindricas
  Dist_gals=[] #galaxias contidas no raio
  densi_grad=[]
  densi_dist=[]
  densi_distS=[]
  densi_distI=[]
  idx_ant=[]
  for q in range(len(dist_from_fil)):
    M=dist_from_fil[q]
    d=Mpc2deg(conv,M)
    aux_idx = ptos_within(ptos_orig, ptos_fil, d)
    idx = np.setdiff1d(aux_idx,idx_ant)
    idx_ant=aux_idx

    rows_grad=dt_gals.iloc[idx]
    rows_dist=dt_gals.iloc[aux_idx]

    densi_grad.append(len(idx)/(compri_mpc*0.5*2))
    densi_dist.append(len(aux_idx)/(compri_mpc*M*2))
    densi_distS.append((len(aux_idx)*compri_mpcS)/(compri_mpc*compri_mpc*M*2))
    densi_distI.append((len(aux_idx)*compri_mpcI)/(compri_mpc*compri_mpc*M*2))

    Grad_gals.append(rows_grad)
    Dist_gals.append(rows_dist)

  #densidade do campo ------------------------------------------------------------------------
  #area em (mpc/h)^2
  # area_field = area_ellipse(conv, np.asarray(orig['ra']), np.asarray(orig['dec']))/(h*h)
  area_field = area_ellipse(conv, np.asarray(dt_gals['ra']), np.asarray(dt_gals['dec']))
  densi_field = np.size(dt_gals['ra'])/area_field #num_gal/area (mpc^2)

  #densidade relativa ------------------------------------------------------
  densi_rlt_grad=[]
  for q in range(len(dist_from_fil)):
    densi_rlt_grad.append(densi_grad[q]/densi_field)

  densi_rlt_dist=[]
  densi_rlt_distS=[]
  densi_rlt_distI=[]
  for q in (range(len(dist_from_fil))):
    densi_rlt_dist.append(densi_dist[q]/densi_field)
    densi_rlt_distS.append(densi_distS[q]/densi_field)
    densi_rlt_distI.append(densi_distI[q]/densi_field)

  return Grad_gals, densi_grad, densi_rlt_grad, Dist_gals, densi_dist, densi_rlt_dist, densi_rlt_distS, densi_rlt_distI
#****************************

def Slices(dt_gals,ra_fil, dec_fil,conv, densi_field):

  n_slc=10
  ra_slc=np.array_split(ra_fil,n_slc)
  dec_slc=np.array_split(dec_fil,n_slc)
  ptos_orig=np.array([dt_gals['ra'],dt_gals['dec']]).T


  Gals_Slice=[]
  dslc=pd.DataFrame(columns=['CompriSlice_Mpc', 'DensRltSlice'])
  #faz as contas para cada fatia
  for f in range(n_slc):
    ptos_slc_fil = np.column_stack((ra_slc[f],dec_slc[f]))
    M=1. #numero de galaxias mais proximas do que 1Mpc/h ao eixo do filamento
    d=Mpc2deg(conv,M)
    in_idx_slc=ptos_within(ptos_orig, ptos_slc_fil, d)
    rows=dt_gals.iloc[in_idx_slc]
    #calcula valores importantes----------------------------------------
    #fita subarray
    p1=scp.interpolate.interp1d(ra_slc[f],dec_slc[f])
    xp = np.linspace(min(ra_slc[f]), max(ra_slc[f]),100)
    aj = p1(xp)

    compri_slc = arc_length(xp, aj)
    compri_mpc_slc = deg2Mpc(conv,compri_slc)
    densi_slc = np.size(in_idx_slc)/(compri_mpc_slc*M*2)
    densi_rlt_slc = densi_slc/densi_field

    dslc=dslc.append({'CompriSlice_Mpc' : compri_mpc_slc, 'DensRltSlice':densi_rlt_slc}, ignore_index=True)
    Gals_Slice.append(rows)


  return Gals_Slice, dslc
#****************************

def dist2Fil_Cl(dt_gals,ptos_fil,ptos_excl,conv):
  aux_degfil=[]
  aux_mpcfil=[]
  aux_degcl=[]
  aux_mpccl=[]
  ra=dt_gals['ra'].values
  dec=dt_gals['dec'].values
  for d in range(len(dt_gals)):
    xf=ra[d]
    yf=dec[d]
    #distancia ao filamento
    aux_deg, aux_mpc=distancia_Mpc(xf, yf, ptos_fil, conv)
    aux_degfil.append(aux_deg)
    aux_mpcfil.append(aux_mpc)

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

  return aux_degfil, aux_mpcfil, aux_degcl, aux_mpccl
#****************************

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# MAIN
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def AnalFil(dt_g,ang,lin, conv):


  in_data='f1.csv'
  out_name='f1'
  excl_cls='real_cls_in_fils.csv'
  fil_init='f1_init.csv'
  fil_tot='realfil06.txt'

  excl=pd.read_csv(excl_cls, delimiter=',')
  ptos_excl = excl.as_matrix(columns=excl.columns[1:3])

  #limpa dados usando o ajuste da RS
  # orig=pd.read_csv(in_data_orig, delimiter=',')
  orig=dt_g
  gr_sup=ang*(orig['Mr'].values) + lin +0.15
  dTot=orig.query('(-24.5 < Mr < -18) & (-0.5 < gr < @gr_sup)')
  dTot_Dirty = orig.copy()

  #ordena pontos do filamento
  # ft=pd.read_csv(fit_tot,delimiter=',',header=None)
  # fil_data=pd.read_csv(in_data, delimiter=',')
  # raN,decN = ProjCorrFil_unico(dt_g.ra.values,dt_g.dec.values,rt[0].values,rt[1].values,fil_data.col1.values,fil_data.col2.values)
  # fil_data.col1 = raN
  # fil_data.col2 = decN

  # fil_0=pd.read_csv(fil_init, delimiter=',', header=None)
  # raN,decN = ProjCorr(fil_init[0].values,fil_init[1].values)
  # fil_init[0] = raN
  # fil_init[1] = decN

  ra_fil,dec_fil=ordena_fil(fil_init, in_data)
  ptos_fil=np.array([ra_fil,dec_fil]).T
  plt.scatter(np.arange(0,len(ra_fil),1),ra_fil)
  plt.show()


  #cria sample limpo de clusters
  #acha indices que pertencem à região do aglomerado
  # idx_InCl = index_GalsinCluster(excl_cls,ptos_fil,dTot,conv)


  # dClean=dTot.copy()
  # dClean=dClean.drop(dClean.index[idx_InCl])
  dTot_Dirty = dTot_Dirty.query('ClusterMember != 1') #sem considerar cortes de magnitude
  dClean = dTot.query('ClusterMember != 1')
  dClean.reset_index(drop=True,inplace=True)
  ptos_clean = dClean.as_matrix(columns=dClean.columns[0:2])
  #calcula luminosidade 


  #Calcula comprimento e densidade do filamento. 
  #Salva em 'f1_comrpimento.txt'

  area_field = area_ellipse(conv, np.asarray(dTot.ra), np.asarray(dTot.dec))
  densi_field = len(dTot.ra)/area_field #num_gal/area (mpc^2)
  compri_mpc, compri_mpcLimSup, compri_mpcLimInf=fil_properties(ra_fil, dec_fil,conv,area_field, len(dTot.ra))

  #Analise do filamento SEM EXCLUIR AGLOMERADOS
  #GRADIENTE
  Grad_gals, densi_grad, densi_rlt_grad, Dist_gals, densi_dist, densi_rlt_dist, ig1,ig2 = Gradiente(dTot, ptos_fil,compri_mpc, compri_mpcLimSup, compri_mpcLimInf,conv)
  #FATIAS
  Slice_gals, vals_slices = Slices(dTot,ra_fil, dec_fil,conv, densi_field)

  #Analise do filamento EXCLUINDO AGLOMERADOS
  #GRADIENTE
  Grad_gals_c, densi_grad_c, densi_rlt_grad_c, Dist_gals_c, densi_dist_c, densi_rlt_dist_c, densi_rlt_distS_c, densi_rlt_distI_c  = Gradiente(dClean, ptos_fil,compri_mpc, compri_mpcLimSup, compri_mpcLimInf,conv)
  Grad_gals_Dc, densi_grad_Dc, densi_rlt_grad_Dc, Dist_gals_Dc, densi_dist_Dc, densi_rlt_dist_Dc, ig1, ig2 = Gradiente(dTot_Dirty, ptos_fil,compri_mpc, compri_mpcLimSup, compri_mpcLimInf,conv)
  #FATIAS
  Slice_gals_c, vals_slices_c = Slices(dClean,ra_fil, dec_fil,conv, densi_field)


  #TODAS AS GALAXIAS DO FILAMENTO
  GalsinFil=Dist_gals[len(Dist_gals)-1].copy()
  degfil,mpcfil,degcl,mpccl= dist2Fil_Cl(GalsinFil,ptos_fil,ptos_excl,conv)
  GalsinFil['Dist2Fil_deg']=degfil
  GalsinFil['Dist2Fil_Mpc']=mpcfil
  GalsinFil['Dist2ClosCluster_deg']=degcl
  GalsinFil['Dist2ClosCluster_Mpc']=mpccl

  GalsinFil_c=Dist_gals_c[len(Dist_gals_c)-1].copy()
  degfil,mpcfil,degcl,mpccl= dist2Fil_Cl(GalsinFil_c,ptos_fil,ptos_excl,conv)
  GalsinFil_c['Dist2Fil_deg']=degfil
  GalsinFil_c['Dist2Fil_Mpc']=mpcfil
  GalsinFil_c['Dist2ClosCluster_deg']=degcl
  GalsinFil_c['Dist2ClosCluster_Mpc']=mpccl

  GalsinFil_Dc=Dist_gals_Dc[len(Dist_gals_Dc)-1].copy()

  # adiciona coluna dizendo qual a distancia maxima que a galaxia pertence
  dist_from_fil=[0.25,0.5,0.75,1.,1.25,1.5]
  for q in range(len(dist_from_fil)):
    if q ==0:
      lim=dist_from_fil[q]
      if len(GalsinFil_c.loc[GalsinFil_c.Dist2Fil_Mpc <= lim]) > 0:
        GalsinFil_c.loc[GalsinFil_c.Dist2Fil_Mpc <= lim, 'GradSlice_MaxDist'] = lim
        GalsinFil.loc[GalsinFil.Dist2Fil_Mpc <= lim, 'GradSlice_MaxDist'] = lim
    else:
      lim_inf=dist_from_fil[q-1]
      lim_sup=dist_from_fil[q]
      tt= GalsinFil_c.loc[ (lim_inf < GalsinFil_c.Dist2Fil_Mpc) & (GalsinFil_c.Dist2Fil_Mpc <= lim_sup)]
      if len(tt)> 0:
        GalsinFil_c.loc[ (lim_inf < GalsinFil_c.Dist2Fil_Mpc) & (GalsinFil_c.Dist2Fil_Mpc <= lim_sup) , 'GradSlice_MaxDist'] = lim_sup
        GalsinFil.loc[ (lim_inf < GalsinFil.Dist2Fil_Mpc) & (GalsinFil.Dist2Fil_Mpc <= lim_sup) , 'GradSlice_MaxDist'] = lim_sup

  # ANALISE CDFs
  gr_T=(dClean.Mg - dClean.Mr).values
  gr_fil=(GalsinFil_c.Mg - GalsinFil_c.Mr).values
  Mg_fil=GalsinFil_c.Mg.values
  Mr_fil=GalsinFil_c.Mr.values
  dist2fil=GalsinFil_c.Dist2Fil_Mpc.values
  dist2cl=GalsinFil_c.Dist2ClosCluster_Mpc.values

  #usando vale do campo:
  print('')
  print('Análise de cor das galáxias')
  print('com limite entre azuis e vermelhas')
  print('dada por todas as galáxias do campo')
  green_val=anal_cor_fil(gr_fil ,Mg_fil, Mr_fil, gr_T, dist2fil, dist2cl, 'gi_campo', 'FIELD')

  # salva tudo
  # GALAXIAS GRAD
  GalsinFil.to_csv('GalsInFil_Grad_Dirty.csv')
  GalsinFil_c.to_csv('GalsInFil_Grad_Clean.csv')
  GalsinFil_Dc.to_csv('GalsInFil_noCl_noColorCut.csv')

  #densidade grad
  c1=['0-0.5','0.5-1.0','1.0-1.5']
  c2=['0.5','1.0','1.5']
  cols=['RangeGrad (Mpc)', 'DensiRelGrad_Dirty','DensiRelGrad_Clean', 'DistMax2Fil (Mpc)', 'DensiRelDist_Dirty','DensiRelDist_Clean','DensiRelDist_LimSup_Clean','DensiRelDist_LimInf_Clean']

  dg=pd.DataFrame(columns=cols)
  dg['RangeGrad (Mpc)']=c1
  dg['DensiRelGrad_Dirty']=densi_rlt_grad
  dg['DensiRelGrad_Clean']=densi_rlt_grad_c
  dg['DistMax2Fil (Mpc)']=c2
  dg['DensiRelDist_Dirty']=densi_rlt_dist
  dg['DensiRelDist_Clean']=densi_rlt_dist_c
  dg['DensiRelDist_LimSup_Clean']=np.array(densi_rlt_distS_c) - np.array(densi_rlt_dist_c)
  dg['DensiRelDist_LimInf_Clean']=np.array(densi_rlt_dist_c) - np.array(densi_rlt_distI_c)
  dg.to_csv('GalsInFil_Grad_DensiRel.csv')

  #GALAXIAS SLICE
  for n in range(len(Slice_gals)):
    Slice_gals[n]['Group'] = n+1
    Slice_gals_c[n]['Group'] = n+1

  Slice_gals_f=pd.concat(Slice_gals,ignore_index=True)
  Slice_gals_cf=pd.concat(Slice_gals_c,ignore_index=True)
  Slice_gals_f.to_csv('GalsInFil_Slice_Dirty.csv')
  Slice_gals_cf.to_csv('GalsInFil_Slice_Clean.csv')

  #densidade slice
  c1=np.arange(1,len(Slice_gals)+1, 1)
  cols=['Group', 'Lenght_Dirty (Mpc)','DensiRel_Dirty', 'Lenght_Clean (Mpc)', 'DensiRel_Clean']

  ds=pd.DataFrame(columns=cols)
  ds['Group']=c1
  ds['Lenght_Dirty (Mpc)']=vals_slices['CompriSlice_Mpc'].values
  ds['DensiRel_Dirty']=vals_slices['DensRltSlice'].values
  ds['Lenght_Clean (Mpc)']=vals_slices_c['CompriSlice_Mpc'].values
  ds['DensiRel_Clean']=vals_slices_c['DensRltSlice'].values

  return green_val

