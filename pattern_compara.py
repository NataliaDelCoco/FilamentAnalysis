#Natalia del Coco
#Maio/2019
#========================================
#plota as patterns detectadas
#=======================================
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
from projection_radec_to_xy import proj
#=======================================
def Mpc2arc(conv,Mpc):
  kpc=Mpc*1000
  arcmin=(kpc/60)/conv
  return arcmin

def arc2Mpc(conv,arc):
  Mpc=arc*conv*60/1000
  return Mpc

#********* ENTRADAS ********************
cl_indt='real_cls_in_fils.csv'
fil_indt6='realfil06.txt'
fil_iner6='errorfil06.csv'
fil_indt7='realfil.txt'
fil_iner7='errorfil.csv'
fil_indt8='realfil.txt'
fil_iner8='errorfil.csv'



in_orig='data_med_mapa_n_v2.csv'
f1_in='f1.csv'
conv=np.float(sys.argv[1])
#--------LE ENTRADAS - Clusters--------------
print('')
print('Le entradas...')


cl=pd.read_csv(cl_indt, delimiter=',')
clz = np.asarray(cl['Redshift']) #redshift
clra = np.asarray(cl['RA'])
cldec = np.asarray(cl['DEC'])

#--------LE ENTRADAS - filamentos--------------
fil_dt6=pd.read_csv(fil_indt6, delimiter=',', header=None)
filra6=fil_dt6[0]
fildec6=fil_dt6[1]
ptos_fil_arc6=np.asarray((filra6,fildec6)).T
filra6=np.asarray(filra6)
fildec6=np.asarray(fildec6)

fil_erf6=pd.read_csv(fil_iner6, delimiter=',')
fil_er6=fil_erf6['Erro']
fil_estab6=fil_erf6['Estabilidade']
fil_rej6=fil_erf6['Rejeita']


path='/home/natalia/Dados/filamentos/SCMS/A1758/old/'

fil_dt7=pd.read_csv(path+'07_08/'+fil_indt7, delimiter=',', header=None)
filra7=fil_dt7[0]
fildec7=fil_dt7[1]
ptos_fil_arc7=np.asarray((filra7,fildec7)).T
filra7=np.asarray(filra7)
fildec7=np.asarray(fildec7)

fil_erf7=pd.read_csv(path+'07_08/'+fil_iner7, delimiter=',')
fil_er7=fil_erf7['Erro']
fil_estab7=fil_erf7['Estabilidade']
fil_rej7=fil_erf7['Rejeita']

fil_dt8=pd.read_csv(path+'08_09/'+fil_indt8, delimiter=',', header=None)
filra8=fil_dt8[0]
fildec8=fil_dt8[1]
ptos_fil_arc8=np.asarray((filra8,fildec8)).T
filra8=np.asarray(filra8)
fildec8=np.asarray(fildec8)

fil_erf8=pd.read_csv(path+'08_09/'+fil_iner8, delimiter=',')
fil_er8=fil_erf8['Erro']
fil_estab8=fil_erf8['Estabilidade']
fil_rej8=fil_erf8['Rejeita']

#--------LE ENTRADAS - original--------------
orig=pd.read_csv(in_orig, delimiter=',')
ra_orig= np.asarray(orig['ra'])
dec_orig = np.asarray(orig['dec'])



#********* MAIN ********************

r_dec_arc=((dec_orig.max()-dec_orig.min())/2)*60
r_ra_arc=((ra_orig.max()-ra_orig.min())/2)*60

#acha centro da elipse:
ra0=ra_orig.max()-(r_ra_arc/60)
dec0=dec_orig.max()-(r_dec_arc/60)



print('')
print('Começa o plot...')

fig=plt.figure()
ax1=fig.add_subplot(111)


#-----pontos originais
ax1.scatter(ra_orig,dec_orig, marker='.', color='lightslategrey', s=5, zorder=2, alpha=0.5)
print('')
print('Plot filamentos...')

#-------plota filamentos ----
# 06
c6=[0.09019608, 0.74509804, 0.81176471, 1.        ]
ax1.scatter(filra6,fildec6,marker='.', color=c6, zorder=2, s=8)

c7=[0.89019608, 0.46666667, 0.76078431, 1.        ]
ax1.scatter(filra7,fildec7,marker='.', color=c7, zorder=2, s=8)

# c8=[0.7372549 , 0.74117647, 0.13333333, 1.        ]
c8=[0.7372549 , 0.74117647, 0.13333333, 1.        ]
ax1.scatter(filra8,fildec8,marker='.', color=c8, zorder=2, s=8)



plt.xlabel('RA',fontsize='large', fontstyle='italic')
plt.ylabel('DEC',fontsize='large', fontstyle='italic')

legend_elements = [Line2D([0], [0], marker='o', color='w', label=r'$A_0 = 0.6$', markerfacecolor=c6, markersize=7),
                  Line2D([0], [0], marker='o', color='w', label=r'$A_0 = 0.7$', markerfacecolor=c7, markersize=7),
                  Line2D([0], [0], marker='o', color='w', label=r'$A_0 = 0.8$', markerfacecolor=c8, markersize=7),
                                 ]


# ax1.legend([ellip3], [r'$r = 8 H^{-1}$ Mpc'])
# ax1.legend([ellip4], [r'$r = 20$ Mpc'])
ax1.legend(handles=legend_elements, loc='lower left',fancybox=True, framealpha=0.3, fontsize='medium')

ax1.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.1f}°"))
ax1.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.1f}°"))
fig.set_size_inches(8.5, 8.5)
plt.gca().invert_xaxis()
plt.savefig('patterns_compara.png')

#mesmo plot, so que com zoom

xlim=plt.xlim()
xl1=ra0+(xlim[0]-ra0)*0.2
xl2=ra0+(xlim[1]-ra0)*0.2
ylim=plt.ylim()
yl1=dec0+(ylim[0]-dec0)*0.2
yl2=dec0+(ylim[1]-dec0)*0.2
# if ylim[0]>0:
#   yl1=dec0+(ylim[0]-dec0)*0.2

# if ylim[0] > 0:
#   yl2=dec0+(ylim[1]-dec0)*0.2


ax1.set_xlim(xl1,xl2)
ax1.set_ylim(yl1,yl2)


plt.savefig('patterns_final2_zoom.png')
plt.close()

