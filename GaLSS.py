#Natalia Del Coco
#Outubro/19

# DEVE SER EXECUTADO NO DIRETORIO DO FILAMENTO
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import sys
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import astropy.units as un
from scipy import  integrate
import pandas as pd
#==================
from filtraZ import filtra_z
from RedSeq import fit_RedSeq
from KDE_RS_V2 import KDE_RSfit
from FilAnalise import AnalFil
from AglAnalise import AnalCl
from ProjCorr import ProjCorr


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ENTRADAS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cluster=sys.argv[1]
conv=float(sys.argv[2])
sofiltra=int(sys.argv[3]) #sofiltra=0 => faz tudo



path_agl='/home/natalia/Dados/Aglomerados/Xdata_total.csv'
path_SCMS='/home/natalia/Dados/filamentos/SCMS/'
path_cl=path_SCMS+cluster


file_orig=path_cl+'/'+cluster+'_f_Natalia_Del_Coco.csv'

f_agl=pd.read_csv(path_agl,delimiter=',')
row = f_agl.loc[f_agl['Cluster'] == cluster]
z_ag = row[' redshift'].values[0]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# FILTRA ARQUIVO INICIAL, CALCULA CORREÇÃO K E MAGS ABSOLUTAS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# filtra_z(file_orig, z_ag)

#saidas: "data_med_mapa_n_v2.csv" => usada aqui
#        "data_med_mapa_n_2_v2.txt" => detecção SCMS

if sofiltra == 0:

  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # SEQUENCIA VERMELHA DOS AGLOMERADOS
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  fg=path_cl+'/'+'data_med_mapa_n_v2.csv'
  fcl=path_cl+'/'+'real_cls_in_fils.csv'
  dt_g=pd.read_csv(fg,delimiter=',')
  dt_cl=pd.read_csv(fcl,delimiter=',')

  #corrige da projeção
  # raN,decN = ProjCorr(dt_g.ra.values,dt_g.dec.values)
  # dt_g.ra=raN
  # dt_g.dec=decN

  dt_g['ClusterMember'] = 0

  #Para cada aglomerado no filamento, acha a RS.
  #O primeiro é o mais importanto, estudado em R-X
  for n in range(len(dt_cl)):
    if n == 0:
      outname = 'GalsInCluster'
    else:
      outname = 'GalsInCluster' + str(n)
    dt_g = KDE_RSfit(dt_g, dt_cl.iloc[n],outname)


  #acha galaxias que pertencem à RS:
  # KDE_RSfit(cluster,f, path_agl)

  #ajusta RS CLUSTER
  f_clean=path_cl+'/'+'GalsInCluster_clean.csv'
  dt=pd.read_csv(f_clean,delimiter=',')
  outname=cluster+'_RS_Cluster.png'
  ang_cl,lin_cl,ang_clErr,lin_clErr=fit_RedSeq(dt,outname) #devolve os parametros angular e linear do ajuste da RS

  #Cria arquivo final com as galaxias do aglomerado CENTRAL
  AnalCl(dt,ang_cl,lin_cl, conv)

  #ajusta RS CAMPO
  outname=cluster+'_RS_Field.png'
  ang_field,lin_field,ang_fieldErr,lin_fieldErr=fit_RedSeq(dt_g,outname) #devolve os parametros angular e linear do ajuste da RS

  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # FILAMENTO
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # estuco concentrado em filamento principal
  # usa ajuste da RS para excluir galáxias de background

  gv=AnalFil(dt_g,ang_cl,lin_cl, conv)

  #ajusta sequencia vermelha do filamento

  f_fil=path_cl+'/'+'GalsInFil_Grad_Clean.csv'
  dt=pd.read_csv(f_fil,delimiter=',')
  outname=cluster+'_RS_Fil.png'
  ang_fil,lin_fil,ang_filErr,lin_filErr=fit_RedSeq(dt,outname)
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # SALVA VALORES RS E GV
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  cols=['Ang_RS_cluster', 'ZeroPoint_RS_cluster', 'Ang_RS_clusterErr', 'ZeroPoint_RS_clusterErr','Ang_RS_field', 'ZeroPoint_RS_field','Ang_RS_fil', 'ZeroPoint_RS_fil', 'GreenValley_field (g-r)']
  data=[[ang_cl,lin_cl,ang_clErr,lin_clErr, ang_field, lin_field, ang_fil, lin_fil, gv]]
  df=pd.DataFrame(data,columns=cols)
  df.to_csv('RS_fit_Total.csv')

  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  # PLOTA PATTERNS FINAL
  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
