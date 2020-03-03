# Natalia Del Coco    19 agosto 2019
#
# Calcula a PDF de uma distribuicao
# verifica se é gaussiana ou não
# acha máximos (até 2, para bimodais) e vales (até 1, para bimodais)
# plota a distribuição, a pdf e ptos críticos. 

#===========================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde as g_kde
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

#===========================================================

def getExtremePoints(data, typeOfInflexion = None, maxPoints = None):
    """
    This method returns the indeces where there is a change in the trend of the input series.
    typeOfExtreme = None returns all extreme points, max only maximum values and min
    only min,
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange ==1)[0]
    if typeOfInflexion == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]
        
    elif typeOfInflexion == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]
    elif typeOfInflexion is not None:
        idx = idx[::2]
    
    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data)-1) in idx:
        idx = np.delete(idx, len(data)-1)
    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx= idx[:maxPoints]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data)//(maxPoints + 1))
    
    return idx

#===========================================================
# MAIN
#===========================================================

def distribution1(data, x_title, output_name):
  if type(data) == type(list()): 
    x=np.asarray(data)[:, np.newaxis]
  else:
    x = data[:, np.newaxis]

  # cria grid pra calcular a pdf
  x_plot=np.linspace(min(x), max(x), 500)[:, np.newaxis]
  binis=np.linspace(min(x), max(x), 50)

  # busca pelo parâmetro 'bandwidth' que gera a melhor pdf
  grid = GridSearchCV(KernelDensity(kernel = 'gaussian'),{'bandwidth': np.linspace(0.01, 0.5, 20)}, cv = 20, iid = True)
  grid.fit(x)
  best_bandw=str(grid.best_params_)
  best_bandw=float(best_bandw.split(': ')[1].split('}')[0])
  print (grid.best_params_)

  # Kernel density Estimation 
  kde = KernelDensity(kernel='gaussian', bandwidth=best_bandw).fit(x)
  log_dens = kde.score_samples(x_plot)

  # avalia se eh 1 gaussiana ou nao
  # hipotese nula: x vem de uma distribuicao gaussiana
  # verifica pelo teste p
  k2, p = stats.normaltest(x)
  alpha = 1e-3

  print('\n\n')
  if p < alpha:  # null hypothesis: x comes from a normal distribution
       print("The null hypothesis can be rejected within %.3f%% " % (alpha))
       bimod=1
  else:
       print("The null hypothesis cannot be rejected")
       bimod=0


  plt.hist(data, bins=15, density=True, color='lightcoral', alpha=0.9, rwidth=0.8, zorder=0)
  plt.plot(x_plot, np.exp(log_dens), linewidth=3, alpha=0.8, zorder=5, label='PDF')
  plt.show()
  print('\n\n')
  maximos=int(input('Quantos máximos quer buscar? 1, 2 ou 3?  '))
  minimos=int(input('Quantos mínimos quer buscar? 0, 1 ou 2?  '))
  plt.close()

  if (bimod == 1) or (maximos > 1):
    modes=getExtremePoints(np.exp(log_dens), 'max', maximos)
    vales=getExtremePoints(np.exp(log_dens), 'min', minimos)

    print('\n\n')
    print('Os valores dos máximos são:')
    for d in range(len(modes)):
      aux = x_plot[modes[d]][0]
      print(aux)
    print('')
    print('e os mínimos:')
    for d in range(len(vales)):
      aux = x_plot[vales[d]][0]
      print(aux)
    print('')

    #vizualiza
    plt.hist(data, bins=15, density=True, color='lightcoral', alpha=0.9, rwidth=0.8, zorder=0)
    plt.plot(x_plot, np.exp(log_dens), linewidth=3, alpha=0.8, zorder=5, label='PDF', color='royalblue')
    plt.scatter(x_plot[modes], np.exp(log_dens[modes]), color='olivedrab', marker='^', s=100, zorder=10, label = r'$PDF_{max}$')
    plt.scatter(x_plot[vales], np.exp(log_dens[vales]), color='olivedrab', marker='v', s=100, zorder=10, label = r'$PDF_{min}$')
    plt.show()
    first_mode = float(input('Qual o valor do primeiro pico de interesse?  '))
    sec_mode = float(input('Qual o valor do segundo pico de interesse?  '))
    valley = float(input('Qual o valor do vale de interesse?  '))
    plt.close()



    # first_mode=x_plot[modes[0]][0]
    # sec_mode=x_plot[modes[1]][0]
    # valley=x_plot[vales[0]][0]

    head1=('p-value, first_mode, second_mode, valley')
    val=[p[0], first_mode, sec_mode, valley]
    out_tab=output_name+'values.csv'
    with open(out_tab,'w') as t:
      t.write(head1)
      t.write("\n")
      for y in range(len(val)):
        v=str(val[y])
        t.write(v)
        if (y < (len(val)-1)): 
          t.write(", ")
    t.close()

  else:
    modes=getExtremePoints(np.exp(log_dens), 'max', 1)
    first_mode=x_plot[modes[0]][0]

    head1=('p-value, mode')
    val=[p[0], first_mode]
    out_tab=output_name+'values.csv'
    with open(out_tab,'w') as t:
      t.write(head1)
      t.write("\n")
      for y in range(len(val)):
        v=str(val[y])
        t.write(v)
        if (y < (len(val)-1)): 
          t.write(", ")
    t.close()


  plt.hist(data, bins=15, density=True, color='lightcoral', alpha=0.9, rwidth=0.8, zorder=0)
  plt.plot(x_plot, np.exp(log_dens), linewidth=3, alpha=0.8, zorder=5, label='PDF', color='royalblue')

  plt.scatter(x_plot[modes], np.exp(log_dens[modes]), color='olivedrab', marker='^', s=100, zorder=10, label = r'$PDF_{max}$')

  if bimod == 1:
    plt.scatter(x_plot[vales], np.exp(log_dens[vales]), color='olivedrab', marker='v', s=100, zorder=10, label = r'$PDF_{min}$')
  plt.xlim(0,max(data)+0.5)
  plt.xlabel(x_title)
  plt.ylabel('PDF')
  plt.legend(loc='best',fancybox=True, framealpha=0.3, fontsize='medium')
  plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)


  plt.savefig(output_name+'.png')
  plt.close()



#===========================================
def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])


from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


# if "setup_text_plots" not in globals():
#     from astroML.plotting import setup_text_plots
# setup_text_plots(fontsize=8, usetex=True)

#------------------------------------------------------------
# Set up the dataset.
#  We'll create our dataset by drawing samples from Gaussians.

# random_state = np.random.RandomState(seed=1)

# X = np.concatenate([random_state.normal(-1, 1.5, 350),
#                     random_state.normal(0, 1, 500),
#                     random_state.normal(3, 0.5, 150)]).reshape(-1, 1)

def distribution(data, x_title, output_name):
  N = np.arange(2, 3)
  models = [None for i in range(len(N))]

  data_shape = np.array(data).reshape(-1,1)
  for i in range(len(N)):
      models[i] = GaussianMixture(N[i]).fit(data_shape)

  # compute the AIC and the BIC
  AIC = [m.aic(data_shape) for m in models]
  BIC = [m.bic(data_shape) for m in models]


  # plot 1: data + best-fit mixture

  M_best = models[np.argmax(AIC)]

  x = np.linspace(min(data_shape)-0.5, max(data_shape)+0.5, 1000)
  logprob = M_best.score_samples(x.reshape(-1, 1))
  responsibilities = M_best.predict_proba(x.reshape(-1, 1))
  pdf = np.exp(logprob)
  pdf_individual = responsibilities * pdf[:, np.newaxis] #tem as duas dists

  #primeira PDF:
  pdf1=pdf_individual.T[0]
  max_pdf1 = np.argmax(pdf1)
  x1_max = x[max_pdf1]
  mean_pdf1 = np.where(pdf1 == pdf1.flat[np.abs(pdf1 - np.mean(pdf1)).argmin()])
  x1_mean = x[mean_pdf1]
  std_pdf1 = np.where(pdf1 == pdf1.flat[np.abs(pdf1 - (max(pdf1) - np.std(pdf1))).argmin()])
  x1_std = x[std_pdf1]

  data1_max=data_shape.flat[np.abs(data_shape - x1_max).argmin()]
  data1_mean=data_shape.flat[np.abs(data_shape - x1_mean).argmin()]
  data1_std = np.abs(data1_max - data_shape.flat[np.abs(data_shape - x1_std).argmin()])

  #segunda PDF:
  pdf2=pdf_individual.T[1]
  max_pdf2 = np.argmax(pdf2)
  x2_max = x[max_pdf2]
  mean_pdf2 = np.where(pdf2 == pdf2.flat[np.abs(pdf2 - np.mean(pdf2)).argmin()])
  x2_mean = x[mean_pdf2]
  std_pdf2 = np.where(pdf2 == pdf2.flat[np.abs(pdf2 - (max(pdf2) - np.std(pdf2))).argmin()])
  x2_std = x[std_pdf2]

  data2_max=data_shape.flat[np.abs(data_shape - x2_max).argmin()]
  data2_mean=data_shape.flat[np.abs(data_shape - x2_mean).argmin()]
  data2_std = np.abs(data2_max - data_shape.flat[np.abs(data_shape - x2_std).argmin()])

#verifica qual eh maior:
  if (data1_max > data2_max):
    aux_max=data1_max
    aux_mean=data1_mean
    aux_std=data1_std

    data1_max=data2_max
    data1_mean=data2_mean
    data1_std=data2_std

    data2_max=aux_max
    data2_mean=aux_mean
    data2_std=aux_std



  #intersecção
  intersec = solve(data1_max,data2_max, data1_std, data2_std)
  print('')
  print('os valores de vale encontrados foram:')
  print('%.6f' % intersec[0])
  print('%.6f' % intersec[1])


  # plt.hist(data_shape, 40, density=True, histtype='stepfilled',color = 'lightcoral', alpha=0.9)
  # plt.plot(x, pdf, color='royalblue', linewidth=3, alpha=0.8)
  # plt.plot(x, pdf_individual, linestyle='--', color='royalblue')
  # if (intersec[0] >= min(data)) and (intersec[0] <= max(data)):
  #   plt.axvline(intersec[0], color='olivedrab', linestyle='-.')
  # if (intersec[1] >= min(data)) and (intersec[1] <= max(data)):
  #   plt.axvline(intersec[1], color='olivedrab', linestyle='-.')

  # plt.show()


  # data_inter = float(input('Entre o valor do vale correto: '))
  data_inter=min(intersec[0],intersec[1])

  # plt.close()
  # if (intersec[0] > data1_max) and (intersec[0] < data2_max):
  #   data_inter = intersec[0]
  # elif (intersec[0] < data1_max) and (intersec[0] > data2_max):
  #   data_inter = intersec[0]
  # elif (intersec[1] > data1_max) and (intersec[1] < data2_max):
  #   data_inter = intersec[1]
  # elif (intersec[1] < data1_max) and (intersec[1] > data2_max):
  #   data_inter = intersec[1]
  # else:
  #   print('')
  #   print('Não encontramos um valor válido para o vale da distribuição')

  #PRINTA
  print('\n\n')
  print('Os valores dos máximos são:')
  print('1st = %.4f | sigma = %0.4f' %(data1_max, data1_std))
  print('2nd = %.4f | sigma = %0.4f' %(data2_max, data2_std))
  print('')
  print('O valor do vale é:')
  print('valley =  %.4f' % data_inter)
  print('')

  #TESTE P => VÊ SE É UMA GAUSSIANA OU NÃO
  k2, p = stats.normaltest(x)
  alpha = 1e-3

  print('\n\n')
  if p < alpha:  # null hypothesis: x comes from a normal distribution
       print("The null hypothesis can be rejected within %.3f%% " % (alpha))
  else:
       print("The null hypothesis cannot be rejected")



#SALVA
  head1=('p-value, 1st_mode, 1st_std, 2nd_mode, 2nd_std, valley')
  val=[p, data1_max, data1_std, data2_max, data2_std, data_inter]
  out_tab=output_name+'_values.csv'
  with open(out_tab,'w') as t:
    t.write(head1)
    t.write("\n")
    for y in range(len(val)):
      v=str(val[y])
      t.write(v)
      if (y < (len(val)-1)): 
        t.write(", ")
  t.close()


  #plota
  ax = plt.subplot(1,1,1)

  ax.hist(data_shape, 40, density=True, histtype='stepfilled',color = 'lightcoral', alpha=0.9)
  ax.plot(x, pdf, color='royalblue', linewidth=3, alpha=0.8)
  ax.plot(x, pdf_individual, linestyle='--', color='royalblue')
  # ax.text(0.04, 0.96, "Best-fit Mixture",
  #         ha='left', va='top', transform=ax.transAxes)
  ax.set_xlabel(x_title)
  # ax.set_ylabel('$p(x)$')

  ax.axvline(data1_max, color='olivedrab', linestyle='-.')
  ax.axvline(data2_max, color='olivedrab', linestyle='-.')
  ax.axvline(data_inter, color='olivedrab', linestyle='-.')
  plt.legend(loc='best',fancybox=True, framealpha=0.3, fontsize='medium')
  plt.xlim(0, 1.75)
  plt.tick_params(direction='in',labelsize=8,top=True, right=True,labeltop=False, labelright=False)
  plt.savefig(output_name+'.png')
  plt.close()

  return data_inter


