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
fil_indt='realfil06.txt'
fil_iner='errorfil06.csv'
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
fil_dt=pd.read_csv(fil_indt, delimiter=',', header=None)
filra=fil_dt[0]
fildec=fil_dt[1]
ptos_fil_arc=np.asarray((filra,fildec)).T
filra=np.asarray(filra)
fildec=np.asarray(fildec)

fil_erf=pd.read_csv(fil_iner, delimiter=',')
fil_er=fil_erf['Erro']
fil_estab=fil_erf['Estabilidade']
fil_rej=fil_erf['Rejeita']

#--------LE ENTRADAS - original--------------
orig=pd.read_csv(in_orig, delimiter=',')
ra_orig= np.asarray(orig['ra'])
dec_orig = np.asarray(orig['dec'])

#--------LE ENTRADAS - f1--------------
f1=pd.read_csv(f1_in, delimiter=',')
#ordena dados pelo ra
f1ra=f1['col1']
f1dec=f1['col2']



#********* MAIN ********************
#Acha r=8 Mpc h-1, considerando as distorções da projeção
# e 1.5 Mpc comovel

# as distorções importam menos na declinação
# regra de três pra achar  a quantos graus correspondem 8 MPC no eixo dec
r_dec_arc=((dec_orig.max()-dec_orig.min())/2)*60
r_dec_Mpc=arc2Mpc(conv,r_dec_arc)
r_dec8_arc=((8/0.7)*r_dec_arc)/r_dec_Mpc
r_dec15_arc=((1.5*(1+clz[0]))*r_dec_arc)/r_dec_Mpc

#acha r=8 mpc/h no eixo RA 
r_ra_arc=((ra_orig.max()-ra_orig.min())/2)*60
r_ra8_arc = (r_dec8_arc*r_ra_arc)/r_dec_arc
r_ra15_arc = (r_dec15_arc*r_ra_arc)/r_dec_arc

#acha centro da elipse:
ra0=ra_orig.max()-(r_ra_arc/60)
dec0=dec_orig.max()-(r_dec_arc/60)

#-------- projeta ------------------
print('')
print('Calcula projeções...')

# transforma as entradas e projeta em x,y (pixel coordinates)
clx, cly = proj(clra, cldec)
filx, fily = proj(filra, fildec)
x_orig,y_orig = proj(ra_orig,dec_orig)



#-------- PLOTA -----------
x_lab=np.linspace(max(filra), min(filra), 7)
x_loc=np.linspace(min(filx), max(filx), 7)
y_lab=np.linspace(min(fildec), max(fildec), 7)
y_loc=np.linspace(min(fily),max(fily), 7)

ult=len(x_lab) - 1

x_lab=np.delete(x_lab,ult)
x_loc=np.delete(x_loc,ult)
y_lab=np.delete(y_lab,ult)
y_loc=np.delete(y_loc,ult)

x_lab=np.delete(x_lab,0)
x_loc=np.delete(x_loc,0)
y_lab=np.delete(y_lab,0)
y_loc=np.delete(y_loc,0)

x_labf=[]
y_labf=[]
for i in range(len(x_lab)):
  x_labf.append(str(round(x_lab[i],1))+'°')
  y_labf.append(str(round(y_lab[i],1))+'°')

print('')
print('Começa o plot...')

fig=plt.figure()
ax1=fig.add_subplot(111)

# plt.xlim(min(filx), max(filx))
# plt.ylim(min(fily), max(fily))

# plt.xlabel('RA',fontsize='large', fontstyle='italic')
# plt.ylabel('DEC',fontsize='large', fontstyle='italic')
# plt.xticks(x_loc,x_labf,fontsize='large')
# plt.yticks(y_loc,y_labf,fontsize='large')

# plt.subplots_adjust(bottom=0.16, right=0.8, top=0.9, left=0.2)


#------ plot area a ser ignorada -------------------
print('')
print('Plot area ignorada...')

# raio_x_max = (max(filra) - min(filra))*1.08
# raio_y_max = (max(fildec) - min(fildec))*1.058
# x_mean=max(filra)-((max(filra) - min(filra))/2)
# y_mean=max(fildec)-((max(fildec) - min(fildec))/2)

raio_x_max=(ra_orig.max()-ra_orig.min())
raio_y_max=(dec_orig.max()-dec_orig.min())
x_mean=max(filra)-((max(filra) - min(filra))/2)
y_mean=max(fildec)-((max(fildec) - min(fildec))/2)

ellip1=Ellipse((ra0,dec0), raio_x_max, raio_y_max,\
 facecolor='darkgray', alpha=0.8, edgecolor='gray')
ellip2 = Ellipse((ra0,dec0), 0.9*raio_x_max, 0.9*raio_y_max,\
 color='white', edgecolor='gray')
# ellip1=Ellipse((clra[0],cldec[0]), raio_x_max, raio_y_max,\
#  facecolor='darkgray', alpha=0.8, edgecolor='white')
# ellip2 = Ellipse((clra[0],cldec[0]), 0.9*raio_x_max, 0.9*raio_y_max,\
#  color='white')

ax1.add_artist(ellip1)
ax1.add_artist(ellip2)

#-----pontos originais
ax1.scatter(ra_orig,dec_orig, marker='.', color='lightslategrey', s=5, zorder=2, alpha=0.3)
print('')
print('Plot filamentos...')

#-------plota filamentos ----
col_fil=['darkgray' if value ==1. else 'gray' for value in fil_rej]
ax1.scatter(filra,fildec,marker='.', color=col_fil, zorder=2, s=8)
ax1.scatter(f1ra,f1dec,marker='.', color='k', zorder=2, s=8)


#-------plota clusters ----
print('')
print('Plot clusters...')


col_cl=['deepskyblue' if value ==0 else 'royalblue' for value in range(len(clx))]
# zd=[10 if value ==0 else 2 for value in range(len(clx))]
s_cl = [90 if value ==0 else 90 for value in range(len(clx))]
ax1.scatter(clra[::-1],cldec[::-1],marker='*', color=col_cl[::-1] ,zorder=2, s=s_cl[::-1], edgecolor='k')

#-------plota raio conectividade ----
print('')
print('Plot raios...')
rxy=r_ra_arc/r_dec_arc

# r20=Mpc2arc(conv,20)/60.
# r20x=r20
# r20y=r20x/rxy

# r8=Mpc2arc(conv,(8/0.7))/60.
# r8x=r8
# r8y=r8x/rxy

ellip3 = Ellipse((clra[0],cldec[0]), r_ra15_arc/60, r_dec15_arc/60, fill=False, color='lightcoral',lw=3)
ellip4 = Ellipse((clra[0],cldec[0]), r_ra8_arc/60, r_dec8_arc/60, fill=False, color='olivedrab',lw=1)

ax1.add_artist(ellip3)
ax1.add_artist(ellip4)

# plt.xlim(max(filra)+0.5,min(filra)-0.5)
# # plt.ylim(min(fildec)-1, max(fildec)+1)
plt.xlabel('RA',fontsize='large', fontstyle='italic')
plt.ylabel('DEC',fontsize='large', fontstyle='italic')
# plt.xticks(x_lab, x_labf,fontsize='large')
# plt.yticks(y_lab, y_labf,fontsize='large')


legend_elements = [Line2D([0], [0], marker='o', color='w', label='Filamento Principal', markerfacecolor='k', markersize=7),
                  Line2D([0], [0], marker='o', color='w', label='Filamento estável', markerfacecolor='gray', markersize=7),
                  Line2D([0], [0], marker='o', color='w', label='Filamento instável', markerfacecolor='lightslategrey', markersize=7, alpha=0.6),
                  Line2D([0], [0], marker='*', color='w', label='AG', markerfacecolor='royalblue', markersize=10),
                  Line2D([0], [0], marker='*', color='w', label='AG central', markerfacecolor='deepskyblue', markersize=10),
                  Line2D([0], [0], marker='o', color='lightcoral', label=r'$r = 1.5$ cMpc', markerfacecolor='w', markersize=7, linestyle=''),
                  Line2D([0], [0], marker='o', color='olivedrab', label=r'$r = 8$ Mpc h$^{-1}$', markerfacecolor='w', markersize=7, linestyle='')
                                 ]


# ax1.legend([ellip3], [r'$r = 8 H^{-1}$ Mpc'])
# ax1.legend([ellip4], [r'$r = 20$ Mpc'])
ax1.legend(handles=legend_elements, loc='lower left',fancybox=True, framealpha=0.3, fontsize='medium')

ax1.yaxis.set_major_formatter(StrMethodFormatter(u"{x:.1f}°"))
ax1.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.1f}°"))
fig.set_size_inches(8.5, 8.5)
plt.gca().invert_xaxis()
plt.savefig('patterns_final2.png')

#mesmo plot, so que com zoom

xlim=plt.xlim()
xl1=ra0+(xlim[0]-ra0)*0.2
xl2=ra0+(xlim[1]-ra0)*0.2
ylim=plt.ylim()
yl1=dec0+(ylim[0]-dec0)*0.2
yl2=dec0+(ylim[1]-dec0)*0.2
ax1.set_xlim(xl1,xl2)
ax1.set_ylim(yl1,yl2)

#-----pontos originais
ax1.scatter(ra_orig,dec_orig, marker='.', color='lightslategrey', s=6, zorder=10, alpha=0.4)





plt.savefig('patterns_final2_zoom.png')
plt.close()

