import numpy as np

def ProjCorr(ra,dec):
  r_ra = (max(ra) - min(ra))/2
  r_dec = (max(dec) - min(dec))/2
  
  aux=[r_ra,r_dec]
  maior=max(aux)
  menor=min(aux)
  dif=maior - menor

  C=dif/max(dec)
  # C=np.sqrt(1-(menor/maior)**2)
  # C=(menor/maior)
  # idx=aux.index(min(aux))


  dec0=max(dec)-r_dec
  ra0=max(ra)-r_ra

  decnew=[]
  for n in range(len(dec)):
    # dist=np.sqrt((dec0 - dec[n])**2 + (ra0 - ra[n])**2)
    dist= dec[n] - dec0
    m=(np.abs(dist)/r_dec)*C
    if dist >= 0:
      decnew.append(dec[n]*(1+m))
    else:
      decnew.append(dec[n]*(1-m))

  return ra,decnew

def ProjCorrFil(ra_cp,dec_cp,ra_fil,dec_fil):
  r_ra = (max(ra_cp) - min(ra_cp))/2
  r_dec = (max(dec_cp) - min(dec_cp))/2
  
  aux=[r_ra,r_dec]
  maior=max(aux)
  menor=min(aux)

  dec0=max(dec_cp)-r_dec
  ra0=max(ra_cp)-r_ra

  r_ra = max(ra_fil) - ra0
  r_dec = max(dec_fil) - dec0
  
  aux=[r_ra,r_dec]
  maior=max(aux)
  menor=min(aux)

  dif=maior - menor


  C=dif/max(dec_fil)
  # C=np.sqrt(1-(menor/maior)**2)
  # C=(menor/maior)
  # idx=aux.index(min(aux))

  decnew=[]

  for n in range(len(dec_fil)):
    # dist=np.sqrt((dec0 - dec[n])**2 + (ra0 - ra[n])**2)
    dist= dec_fil[n] - dec0
    m=(np.abs(dist)/r_dec)*C
    if dist >= 0:
      decnew.append(dec_fil[n]*(1+m))
    else:
      decnew.append(dec_fil[n]*(1-m))


  return ra_fil,decnew


def ProjCorrFil_unico(ra_cp,dec_cp,ra_fil,dec_fil,ra_filU,dec_filU):
  r_ra = (max(ra_cp) - min(ra_cp))/2
  r_dec = (max(dec_cp) - min(dec_cp))/2
  
  aux=[r_ra,r_dec]
  maior=max(aux)
  menor=min(aux)

  dec0=max(dec_cp)-r_dec
  ra0=max(ra_cp)-r_ra

  r_ra = max(ra_fil) - ra0
  r_dec = max(dec_fil) - dec0
  
  aux=[r_ra,r_dec]
  maior=max(aux)
  menor=min(aux)

  dif=maior - menor


  C=dif/max(dec_fil)
  # C=np.sqrt(1-(menor/maior)**2)
  # C=(menor/maior)
  # idx=aux.index(min(aux))

  decnew=[]

  for n in range(len(dec_filU)):
    # dist=np.sqrt((dec0 - dec[n])**2 + (ra0 - ra[n])**2)
    dist= dec_filU[n] - dec0
    m=(np.abs(dist)/r_dec)*C
    if dist >= 0:
      decnew.append(dec_filU[n]*(1+m))
    else:
      decnew.append(dec_filU[n]*(1-m))


  return ra_filU,decnew



def proj(ra,dec,rst):
  x=rst*np.sin(ra)*np.cos(dec)
  y=rst*np.sin(ra)*np.sin(dec)
  z=rst*np.cos(ra)
  return x,y,z