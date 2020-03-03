
from check_distribution import distribution
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx
import matplotlib
import numpy as np
from scipy import stats
import pandas as pd

def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '%.2f' % (x)

# anal_cor_fil(gi_gals_e ,Mg_gals_e, Mi_gals_e, gi_total, dist_galfil_mpc_e, dist_galcl_min, 'gi_fil', 'FIL')

# cor_gi=gi_gals_e
# mag_g=Mg_gals_e
# mag_i=Mi_gals_e
# dist_galfil=dist_galfil_mpc_e
# dist_galcl=dist_galcl_min
# out_name='gi_fil'
# cor_dist='FIL'
# cor_campo=gi_total
# out_name='gi_campo'


def anal_cor_fil(cor_gi,mag_g, mag_i, cor_campo, dist_galfil, dist_galcl, out_name, cor_dist):
 
#"COR-DIST" = 'FIL' => cor para a separação 
#     entre vermelhas e azuis vem da cor do filamento
#"COR-DIST" = 'FIELD' => cor para a separação 
#     entre vermelhas e azuis vem da cor do campo

  x_axis=r'$g - r$'
  if cor_dist == 'FIL':
    data = cor_gi
  else: 
    data = cor_campo
  distribution(data, x_axis, out_name+'_distrib')

  distrib_vals=pd.read_csv(out_name+'_distrib_values.csv', delimiter=',')



  if len(distrib_vals.columns) >= 4:
    bimod=1
    blue_mode = distrib_vals[' 1st_mode'][0]
    red_mode = distrib_vals[' 2nd_mode'][0]
    valley = distrib_vals[' valley'][0]
    print('\n\n')
    print('Distribuição de g - r é bimodal')
    print('Pico azul: g - r = %.5f' % blue_mode)
    print('Pico vermelho: g - r = %.5f' % red_mode)
    print('Vale: g - r = %.5f' % valley)


  # se a distribuicao for bimodal,seguimos com a analise


  dgf_gi=np.vstack((dist_galfil,cor_gi)).T

  #so para vizualização:
  plt.scatter(dist_galfil,cor_gi, color='olivedrab')
  plt.xlabel('Galaxy-filament distance (Mpc)')
  plt.ylabel(r'$g - r$')
  plt.legend(loc='best',fancybox=True, framealpha=0.3, fontsize='medium')
  plt.yscale('log')
  plt.savefig(out_name+'_dist_galfil.png', bbox_inches='tight')
  plt.close()

  blue_gals = []
  red_gals = []
  Mg_blue_gals=[]
  Mg_red_gals=[]
  Mi_blue_gals=[]
  Mi_red_gals=[]
  for d in range(len(cor_gi)):
    if cor_gi[d] <= valley:
      blue_gals.append(dgf_gi[d].tolist())
      Mg_blue_gals.append(mag_g[d].tolist())
      Mi_blue_gals.append(mag_i[d].tolist())
    else:
      red_gals.append(dgf_gi[d].tolist())
      Mg_red_gals.append(mag_g[d].tolist())
      Mi_red_gals.append(mag_i[d].tolist())

  blue_gals = np.array(blue_gals)
  red_gals = np.array(red_gals)
  Mg_blue_gals = np.array(Mg_blue_gals)
  Mi_blue_gals = np.array(Mi_blue_gals)
  Mg_red_gals = np.array(Mg_red_gals)
  Mi_red_gals = np.array(Mi_red_gals)

  #CDF das galáxias em relação à distancia ao eixo do filamento-------
  #Azuis
  H_blue,X1_blue = np.histogram(blue_gals.T[0] , bins = 10, density=True)
  dx = X1_blue[1] - X1_blue[0]
  F1_blue = np.cumsum(H_blue)*dx

  #vermelhas
  H_red,X1_red = np.histogram(red_gals.T[0] , bins = 10, density=True)
  dx = X1_red[1] - X1_red[0]
  F1_red = np.cumsum(H_red)*dx

  #KS test
  # testa a hipotese nula de que as dists são iguais
  KS, p_cdf_galfil = stats.ks_2samp(F1_red, F1_blue)
  alpha=0.1

  print('\n\n')
  print('Teste Kolmogorov–Smirnov:')
  print(' as CDFs das galáxias vermelhas e azuis pela distância ao filamento são iguais?')
  print('p-val = %.3f' %p_cdf_galfil)
  if p_cdf_galfil < alpha:  # null hypothesis: x comes from a normal distribution
       print("The null hypothesis can be rejected within %.3f%% " % (alpha))
       print('As distribuições são diferentes')
  else:
       print("The null hypothesis cannot be rejected")



  #plota CDF----------------------------------------------------------
  formatter = FuncFormatter(log_10_product)

  plt.plot(X1_blue[1:], F1_blue, color='royalblue', label = 'blue galaxies')
  plt.plot(X1_red[1:], F1_red, color='lightcoral', label = 'red galaxies')
  plt.xlabel('Galaxy-filament distance (MPC)')
  plt.ylabel('CDF')
  plt.legend(loc='best',fancybox=True, framealpha=0.3, fontsize='medium')
  plt.yscale('log')
  plt.savefig(out_name+'_dist_galfil_cdf.png', bbox_inches='tight')
  plt.close()

  #FRACAO DE VERMELHAS------------------------------------------------
  #fração de vermelhas media do campo todo e do filamento
  #primeiro acha o vale de cor do campo todo:

  #RED_FRAC FILAMENTO
  n_bin=2
  d_sort = np.array(sorted(dgf_gi, key=lambda x : x[0]))

  bin_dist = np.array(np.array_split(d_sort.T[0], n_bin))
  bin_gi = np.array(np.array_split(d_sort.T[1], n_bin))

  frac_red=[]
  frac_red_err=[]
  dgf_med = []
  dgf_med_err = []
  for d in range(n_bin):
    aux_red=0
    for s in range(len(bin_gi[d])):
      if bin_gi[d][s] > valley:
        aux_red += 1
    print(aux_red)
    print('\n')
    frac_red.append(aux_red/len(bin_gi[d]))
    err=frac_red[d]*np.sqrt(((1/aux_red) + (1/(len(bin_gi[d])))))
    frac_red_err.append(err)
    # frac_blue.append(aux_blue/len(bin_gi))
    # err=frac_blue[d]*np.sqrt(((1/aux_blue) + (1/(len(bin_gi)))))
    # frac_blue_err.append(err)
    dgf_med.append(np.mean(bin_dist[d]))
    dgf_med_err.append(np.std(bin_dist[d]))


  aux_blue=0
  for d in cor_gi:
    if d <= valley:
      aux_blue += 1
  frac_blue=aux_blue/len(cor_gi)
  frac_blue_err = frac_blue*np.sqrt(((1/aux_blue) + (1/(len(cor_gi)))))


  #RED_FRAC CAMPO
  frac_red_campo=[]
  frac_red_campo_err=[]
  aux_red=0
  for s in range(len(cor_campo)):
    if cor_campo[s] > valley:
      aux_red += 1
  frac_red_campo = aux_red/len(cor_campo)
  frac_red_campo_err=frac_red_campo*np.sqrt(((1/aux_red) + (1/(len(cor_campo)))))

  x_fill=np.linspace((min(dgf_med)-0.3),(max(dgf_med)+0.3), 100)

  plt.errorbar(dgf_med, frac_red, xerr=dgf_med_err, yerr=frac_red_err, linestyle='-', marker='o', color='lightcoral', zorder=10)
  plt.hlines(y=frac_red_campo, xmin=min(x_fill), xmax=max(x_fill) , color='dimgrey', label=r'Field red fraction = %.2f $\pm$ %.2f' % (frac_red_campo, frac_red_campo_err), zorder=5, linestyle='--' )
  plt.fill_between(x=x_fill, y1=(frac_red_campo-frac_red_campo_err), y2=(frac_red_campo+frac_red_campo_err), color='dimgrey', alpha=0.3, zorder=0)
  plt.xlabel('Galaxy-filament distance (Mpc)')
  plt.ylabel('Fraction of red galaxies')
  plt.savefig(out_name+'_redfrac_dist_galfil.png', bbox_inches='tight')
  plt.legend(loc='best',fancybox=True, framealpha=0.3, fontsize='medium')
  plt.close()





  # cor magnitude ------------------------

  plt.scatter(Mi_red_gals, red_gals.T[1], color='lightcoral', label='Red galaxies')
  plt.scatter(Mi_blue_gals, blue_gals.T[1], color='royalblue', label='Blue galaxies')
  plt.xlabel(r'$i$')
  plt.ylabel(r'$g - r$')
  plt.legend(loc='best',fancybox=True, framealpha=0.3, fontsize='medium')
  plt.savefig(out_name+'cor_mag_i.png', bbox_inches='tight')
  plt.close()


  #distancia ate os aglomerados----------------------------------
  # n_cl=0
  # dist_galcl_0=[]
  # dist_galcl_min=[]
  # for d in range(len(idx_total_e)):
  #   dist_galcl_0.append(dist_galcl[d][n_cl])
  #   dist_galcl_min.append(np.min(dist_galcl[d]))

  dgc_gi=np.vstack((dist_galcl,cor_gi)).T

  #so para vizualização:
  plt.scatter(dist_galcl,cor_gi, color='olivedrab')
  plt.xlabel('Galaxy-closest cluster distance (Mpc)')
  plt.ylabel(r'$g - r$')
  plt.savefig(out_name+'_dist_galcl.png', bbox_inches='tight')
  plt.close()

  blue_gals = []
  red_gals = []
  Mg_blue_gals=[]
  Mg_red_gals=[]
  Mi_blue_gals=[]
  Mi_red_gals=[]
  for d in range(len(cor_gi)):
    if cor_gi[d] <= valley:
      blue_gals.append(dgc_gi[d].tolist())
      Mg_blue_gals.append(mag_g[d].tolist())
      Mi_blue_gals.append(mag_i[d].tolist())
    else:
      red_gals.append(dgc_gi[d].tolist())
      Mg_red_gals.append(mag_g[d].tolist())
      Mi_red_gals.append(mag_i[d].tolist())

  blue_gals = np.array(blue_gals)
  red_gals = np.array(red_gals)
  Mg_blue_gals = np.array(Mg_blue_gals)
  Mi_blue_gals = np.array(Mi_blue_gals)
  Mg_red_gals = np.array(Mg_red_gals)
  Mi_red_gals = np.array(Mi_red_gals)

  #CDF das galáxias em relação à distancia ao eixo do filamento-------
  #Azuis
  H_blue,X1_blue = np.histogram(blue_gals.T[0] , bins = 20, density=True)
  dx = X1_blue[1] - X1_blue[0]
  F1_blue = np.cumsum(H_blue)*dx

  #vermelhas
  H_red,X1_red = np.histogram(red_gals.T[0] , bins = 20, density=True)
  dx = X1_red[1] - X1_red[0]
  F1_red = np.cumsum(H_red)*dx

  #KS test
  # testa a hipotese nula de que as dists são iguais
  KS, p_val = stats.ks_2samp(F1_red, F1_blue)
  alpha=0.1

  print('\n\n')
  print('Teste Kolmogorov–Smirnov:')
  print(' as CDFs das galáxias vermelhas e azuis pela distância ao filamento são iguais?')
  print('p-val = %.3f' %p_val)
  if p_val < alpha:  # null hypothesis: x comes from a normal distribution
       print("The null hypothesis can be rejected within %.3f%% " % (alpha))
       print('As distribuições são diferentes')
  else:
       print("The null hypothesis cannot be rejected")


  #plota CDF----------------------------------------------------------
  plt.plot(X1_blue[1:], F1_blue, color='royalblue', label = 'blue galaxies')
  plt.plot(X1_red[1:], F1_red, color='lightcoral', label = 'red galaxies')
  plt.xlabel('Galaxy-closest cluster distance (MPC)')
  plt.ylabel('CDF')
  plt.legend(loc='best',fancybox=True, framealpha=0.3, fontsize='medium')
  plt.yscale('log')
  plt.savefig(out_name+'_dist_galcl_cdf.png', bbox_inches='tight')
  plt.close()

  #RED FRACTION
  d_sort = np.array(sorted(dgc_gi, key=lambda x : x[0]))

  bin_dist = np.array(np.array_split(d_sort.T[0], n_bin))
  bin_gi = np.array(np.array_split(d_sort.T[1], n_bin))

  frac_red_gc=[]
  frac_red_gc_err=[]
  dgc_med = []
  dgc_med_err = []
  for d in range(n_bin):
    aux_red=0
    for s in range(len(bin_gi[d])):
      if bin_gi[d][s] > valley:
        aux_red += 1
    print(aux_red)
    print('\n')
    frac_red_gc.append(aux_red/len(bin_gi[d]))
    err=frac_red_gc[d]*np.sqrt(((1/aux_red) + (1/(len(bin_gi[d])))))
    frac_red_gc_err.append(err)
    # frac_blue.append(aux_blue/len(bin_gi))
    # err=frac_blue[d]*np.sqrt(((1/aux_blue) + (1/(len(bin_gi)))))
    # frac_blue_err.append(err)
    dgc_med.append(np.mean(bin_dist[d]))
    dgc_med_err.append(np.std(bin_dist[d]))

  x_fill=np.linspace((min(dgc_med)-0.3),(max(dgc_med)+0.3), 100)

  plt.errorbar(dgc_med, frac_red_gc, xerr=dgf_med_err, yerr=frac_red_gc_err, linestyle='-', marker='o', color='lightcoral', zorder=10)
  plt.hlines(y=frac_red_campo, xmin=min(x_fill), xmax=max(x_fill) , color='dimgrey', label=r'Field red fraction = %.2f $\pm$ %.2f' % (frac_red_campo, frac_red_campo_err), zorder=5, linestyle='--' )
  plt.fill_between(x=x_fill, y1=(frac_red_campo-frac_red_campo_err), y2=(frac_red_campo+frac_red_campo_err), color='dimgrey', alpha=0.3, zorder=0)
  plt.xlabel('Galaxy-closest cluster distance (Mpc)')
  plt.ylabel('Fraction of red galaxies')
  plt.savefig(out_name+'_redfrac_dist_galcl.png', bbox_inches='tight')
  plt.legend(loc='best',fancybox=True, framealpha=0.3, fontsize='medium')
  plt.close()


  out_name_RF=out_name+'_RedFrac.csv'

  salva=pd.DataFrame(columns=['RedFrac_galfil', 'RedFrac_galfil_err', 'Dist_galfil(Mpc)','Dist_galfil_err(Mpc)','RedFrac_galcl', 'RedFrac_galcl_err', 'Dist_galcl(Mpc)','Dist_galcl_err(Mpc)'])
  for q in range(n_bin):
    salva.loc[q] = frac_red[q],frac_red_err[q],dgf_med[q],dgf_med_err[q],frac_red_gc[q],frac_red_gc_err[q],dgc_med[q],dgc_med_err[q]
    salva.to_csv(out_name_RF)


  out_name_BF=out_name+'_colorFrac.csv'
  head1=('Blue_red_separation, BlueFrac_inFil, BlueFrac_inFil_err, RedFrac_field, RedFrac_field_err')
  val=[valley, frac_blue, frac_blue_err, frac_red_campo, frac_red_campo_err]
  with open(out_name_BF,'w') as t:
    t.write(head1)
    t.write("\n")
    for y in range(len(val)):
      v=str(val[y])
      t.write(v)
      if (y < (len(val)-1)): 
        t.write(", ")
  t.close()

  return valley

