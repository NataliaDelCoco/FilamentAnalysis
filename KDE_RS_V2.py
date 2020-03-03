#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:21:56 2019

@author: nilose
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy import stats
import scipy.integrate as integrate

def gauss(x,mu,sigma):
      return (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-0.5*((x-mu)**2)/(sigma**2))
def bigauss(x,mu,sigma, mu2, sigma2):
      return gauss(x,mu,sigma)*gauss(x,mu2,sigma2)

def KDE_RSfit(dt_g,dt_cl,outname):

    xdata = dt_cl
    gals = dt_g
    #referencia = 'A85clean-3,7-4r500.csv'
    r200 = (xdata['R500(arcmin)']/ 60.0 / 0.65)
    rmin = 13.0
    rmax = 23.0
    grmin = -1.0
    grmax = 4.0
    z = xdata['Redshift']
    ra0 = xdata['RA']
    dec0 = xdata['DEC']
    rFin = 4.0*r200
    rFout = 5.0*r200
    rr=40
    if rr == 1:
        rFin = 3.5*r200
        rFout = 3.8*r200
    if rr == 8:
        rFin = 1.3*r200
        rFout = 1.49*r200        
    if rr == 20:
        rFin = 3.0*r200
        rFout = 3.8*r200
    if rr == 30:
        rFin = 5.*r200
        rFout = 5.8*r200
    if rr == 40:
        rFin = 4.*r200
        rFout = 4.8*r200       

    
    areaCL = np.pi * r200**2
    areaF = np.pi * (rFout**2 - rFin**2)
    norm = areaCL / areaF
    
    galsCL = gals.query('(ra - @ra0)**2 + (dec - @dec0)**2 < (@r200)**2 & dered_r < @rmax & dered_r > @rmin & grModelColor < @grmax & grModelColor > @grmin')
    galsF = gals.query('(ra - @ra0)**2 + (dec - @dec0)**2 < (@rFout)**2 & (ra - @ra0)**2 + (dec - @dec0)**2 > (@rFin)**2 & dered_r < @rmax & dered_r > @rmin & grModelColor < @grmax & grModelColor > @grmin')
    
    
    #### Plots the Filed galaxies
    plt.scatter(galsF['ra'], galsF['dec'], marker='o', color='black', s=4)
    nameid =  outname + '-fieldring.png'
    plt.ylabel('DEC (degrees)')
    plt.xlabel('RA (degrees)')
    plt.savefig(nameid, format='png')
    plt.close()
    
    #### Plots the Cluster galaxies
    plt.scatter(galsCL['ra'], galsCL['dec'], marker='o', color='black', s=4)
    nameid =  outname + '-clusterregion.png'
    plt.ylabel('DEC (degrees)')
    plt.xlabel('RA (degrees)')
    plt.savefig(nameid, format='png')
    plt.close()
    
    ####################################
    
    NgalsF = float(len(galsF))
    NgalsCL = float(len(galsCL))
    
    r = galsCL['dered_r']
    gr = galsCL['grModelColor']
    
    xmin = r.min()
    xmax = r.max()
    ymin = gr.min()
    ymax = gr.max()
    
    print(xmin)
    print(xmax)
    print(ymin)
    print(ymax)
    print(norm)
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    
    values = np.vstack([r, gr])
    
    kernelCL = stats.gaussian_kde(values)
    galsCL['kdePDF'] = kernelCL.evaluate(values)
    
    
    ##############################################################################
    ### Field KDE
    
    rField = galsF['dered_r']
    grField = galsF['grModelColor']
    valuesField = np.vstack([rField, grField])
    
    kernelF = stats.gaussian_kde(valuesField)
    galsCL['kdePDFfield'] = kernelF.evaluate(values)       #### KDE PDF do FIELD calculada nos pontos correspondentes as galaxias do Cluster (contaminado)
    
    
    ############################ Probability that a given galaxy is a field galaxy using photoz as prior
    galsCL['prob']=0.0
    galsCL['member']=0.0
    galsCL['prior']=0.0
    meanerror = galsCL['Column3'].std()
    print(meanerror)
    galsclassrest = galsCL.reset_index(drop=True)
    
    # for i in range(len(galsclass1)):
    for i in range(len(galsCL)):
        mu = galsCL['Column2'].values[i]
        sigma = galsCL['Column3'].values[i]
        integral = integrate.quad(gauss, z - 1*meanerror, z + 1*meanerror , args=(mu,sigma))
        prior = 1 - integral[0]
        #integral2 = integrate.quad(bigauss, -np.inf, np.inf , args=(mu,sigma, z, 0.03))    
        #prior2 = 1.0 - integral2[0]
        galsCL['prior'].values[i] = prior
        #galsclass1['prior2'][i] = prior2
        galsCL['prob'].values[i] = norm * galsCL['kdePDFfield'].values[i] * NgalsF / (galsCL['kdePDF'].values[i] * NgalsCL) * prior
        
    galsclassrest['prob'] = norm * galsclassrest['kdePDFfield'] * NgalsF / (galsclassrest['kdePDF'] * NgalsCL)
    
    ##############################################################################
    ####Plotting The dirty KDE
    Z = np.reshape(kernelCL(positions).T, X.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    figure = ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    #ax.plot(r, gr, 'k.', markersize=2)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.scatter(r,gr, marker='.', s=1, color='black')
    cbar = fig.colorbar(figure, ax=ax , use_gridspec=False)
    nameid =  outname + '-dirty.png'
    plt.ylabel('g - r')
    plt.xlabel('r')
    cbar.set_label('PDF')
    plt.savefig(nameid, format='png')
    plt.close()
    #plt.show()
    
    
    #plt.figure()
    
    df = galsCL.copy()
    # df = df.append(galsclass1, ignore_index=True)
    df = df.append(galsclassrest, ignore_index=True)
    
    for m in range(1):
        for i in range(int(len(df))):
            fica=0
            for mcmc in range(100):
                if df['prob'][i] < random.uniform(0,1):
                    fica +=1
            if fica >= 68: #1sigma
                df['member'][i] = 1
                objt=df['obj'][i]
                wh=np.where((gals.ra == df.ra[i]) & (gals.dec == df.dec[i]))[0][0]
                # wh=np.where((gals.obj == objt) ==True)[0]
                gals.ClusterMember[wh]=1
            else:
                df['member'][i] = 0
                wh=np.where((gals.ra == df.ra[i]) & (gals.dec == df.dec[i]))[0][0]
                # wh=np.where((gals.obj == objt) ==True)[0]
                gals.ClusterMember[wh]=2 #indica que nao esta no cluster mas esta em R200
    final=gals.copy()
    clean = df.query('member == 1')
    it = str(m)
    nameid = outname+'_clean.csv'
    clean.to_csv(nameid)
    nameid = outname+'_dirtyWprob.csv'
    df.to_csv(nameid)
    ### Checks normalization of PDFS
        
    kernelCL.integrate_box([-np.inf,-np.inf],[np.inf,np.inf],maxpts=None)
    kernelF.integrate_box([-np.inf,-np.inf],[np.inf,np.inf],maxpts=None)
    
    ############################Plots the Field data plus the fitted KDE
    ZF = np.reshape(kernelF(positions).T, X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    figure = ax.imshow(np.rot90(ZF), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    #ax.plot(rclean, grclean, 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.scatter(rField,grField, marker='.', s=1, color='black')
    cbar = fig.colorbar(figure, ax=ax , use_gridspec=False)
    nameid =  outname+ '-field.png'
    plt.ylabel('g - r')
    plt.xlabel('r')
    cbar.set_label('PDF')
    plt.savefig(nameid, format='png')
    plt.close()
    #plt.show()
    
    
    
    ##################################Plots the clean data plus the fitted KDE
    rclean = clean['dered_r']
    grclean = clean['grModelColor']
    valuesclean = np.vstack([rclean, grclean])
    kernelclean = stats.gaussian_kde(valuesclean)
    Zclean = np.reshape(kernelclean(positions).T, X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    figure = ax.imshow(np.rot90(Zclean), cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    #ax.plot(rclean, grclean, 'k.', markersize=2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.scatter(rclean,grclean, marker='.', s=1, color='black')
    cbar = fig.colorbar(figure, ax=ax , use_gridspec=False)
    nameid =  outname + '-clean.png'
    plt.ylabel('g - r')
    plt.xlabel('r')
    cbar.set_label('PDF')
    plt.savefig(nameid, format='png')
    plt.close()
    #plt.show()
    
    print('##############numeros')
    print('areaCL')
    print(areaCL)
    print('areaF')
    print(areaF)
    print('norm')
    print(norm)
    print('NgalsF')
    print(NgalsF)
    print('NgalsCL')
    print(NgalsCL)
    print('NgalsF*norm')
    print(NgalsF*norm)

    
    
    
    ############################################# Estimador da PDF clean
    # estclean = (np.rot90(Z)*NgalsCL - np.rot90(ZF)*norm*NgalsF)/(NgalsCL - norm*NgalsF)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # figure = ax.imshow(estclean, cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    # #ax.plot(rclean, grclean, 'k.', markersize=2)
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    # #ax.scatter(rclean,grclean, marker='.', s=1, color='black')
    # cbar = fig.colorbar(figure, ax=ax , use_gridspec=False)
    # nameid = outname + '-theoryPDF.png'
    # plt.ylabel('g - r')
    # plt.xlabel('r')
    # cbar.set_label('PDF')
    # plt.savefig(nameid, format='png')
    # plt.close()
    # #plt.show()
    
    
    ############################################# Subtrai a PDF-clean calculada da Sorteada por MC
    # dif = estclean - np.rot90(Zclean)
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # figure = ax.imshow(dif, cmap=plt.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    # #ax.plot(rclean, grclean, 'k.', markersize=2)
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    # #ax.scatter(rclean,grclean, marker='.', s=1, color='black')
    # cbar = fig.colorbar(figure, ax=ax , use_gridspec=False)
    # nameid =  cl + '-theoryPDF-cleanPDF.png'
    # plt.ylabel('g - r')
    # plt.xlabel('r')
    # cbar.set_label('theoretical PDF - clean sample PDF')
    # plt.savefig(nameid, format='png')
    # plt.close()
    return final