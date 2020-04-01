import numpy as np
from numpy import pi,sqrt,exp
from scipy.integrate import simps
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.special import erf
import matplotlib.pyplot as plt
import formats
import pandas as pd
from add_asym import add_asym
"""
Various mathematical routines.
"""

sqrt2pi = sqrt(2.*pi)


def wp(data, wt, percentiles):
  """
  Compute weighted percentiles. 
  If the weights are equal, this is the same as normal percentiles. 
  Elements of the C{data} and C{wt} arrays correspond to 
  each other and must have equal length (unless C{wt} is C{None}). 
  
  @param data: The data.              
  @type data: A L{np.ndarray} array or a C{list} of numbers. 
  @param wt: How important is a given piece of data. 
  @type wt: C{None} or a L{np.ndarray} array or a C{list} of numbers. 
    All the weights must be non-negative and the sum must be 
    greater than zero. 
  @param percentiles: what percentiles to use.  (Not really percentiles, 
    as the range is 0-1 rather than 0-100.) 
  @type percentiles: a C{list} of numbers between 0 and 1. 
  @rtype: [ C{float}, ... ] 
  @return: the weighted percentiles of the data. 
  """ 
  assert np.greater_equal(percentiles, 0.0).all(), "Percentiles less than zero" 
  assert np.less_equal(percentiles, 1.0).all(), "Percentiles greater than one" 
  data = np.asarray(data) 
  assert len(data.shape) == 1 

  if wt is None: 
    wt = np.ones(data.shape, np.float) 
  else:
    wt = np.asarray(wt, np.float)
    assert wt.shape == data.shape
    assert np.greater_equal(wt, 0.0).all(), "Not all weights are non-negative."
    assert len(wt.shape) == 1 
    n = data.shape[0] 
    assert n > 0 
    i = np.argsort(data)
    sd = np.take(data, i, axis=0)
    sw = np.take(wt, i, axis=0)
    aw = np.add.accumulate(sw)

    if not aw[-1] > 0:
      raise ValueError("Nonpositive weight sum")

    w = (aw-0.5*sw)/aw[-1]
    spots = np.searchsorted(w, percentiles)
    o = []

    for (s, p) in zip(spots, percentiles):
      if s == 0:
        o.append(sd[0])
      elif s == n:
        o.append(sd[n-1])
      else:
        f1 = (w[s] - p)/(w[s] - w[s-1])
        f2 = (p - w[s-1])/(w[s] - w[s-1])
        assert f1>=0 and f2>=0 and f1<=1 and f2<=1
        assert abs(f1+f2-1.0) < 1e-6
        o.append(sd[s-1]*f1 + sd[s]*f2)

    return o
#------------------------------------------------------------------------------

def nth_moment(P, x, n, c, normalize=True):
  """
  For a probability density function P(x) defined on x, calculate the n'th
  moment about c.
  This method uses scipy.integrate.simps(), which works best if P is
  moderately well-sampled, I think.
  If keyword 'normalize' is not set to False, P is first normalized.

  Examples:
    Define a PDF:
    >>> x      = np.linspace(-2,10,120)
    >>> xi,w,a = 0,2,6                               #Location, width, asymmetry
    >>> P      = test_add_asym.skewGauss(x,xi,w,a)   #Define PDF
    >>> mu     = test_add_asym.nth_moment(P,x,1,0)   #1st moment about 0 (mean)
    >>> var    = test_add_asym.nth_moment(P,x,2,mu)  #2nd moment about the mean (variance)
    >>> sig    = np.sqrt(var)                        #Standard deviation
    >>> skew   = test_add_asym.nth_moment(P,x,3,mu)/sig**3 #3rd about mean (skewness)
  """
  if normalize: P = P / simps(P,x)
  return simps((x-c)**n * P, x)
#------------------------------------------------------------------------------

def skewness(P, x, normalize=True, Thiele=False):
  """
  For a probability density function P(x) defined on x, calculate the
  skewness, defined by the third moments about the mean.
  If the mean is already at hand, it's faster to use nth_moment().
  If keyword 'normalize' is not set to False, P is first normalized.
  If keyword 'Thiele' is set to True, calculate instead the unnormalized skew,
  which is th third semi-invariant cumulant (Thiele 1889).

  Examples:
    Define a PDF:
    >>> x      = np.linspace(-2,10,120)
    >>> xi,w,a = 0,2,6                             #Location, width, asymmetry
    >>> P      = test_add_asym.skewGauss(x,xi,w,a) #Define PDF
    >>> S      = test_add_asym.skewness(P,x)       #Skewness
  """

  if normalize: P = P / simps(P,x)              #Normalize here rather than at
                                                #every call to nth_moment()
  mu = nth_moment(P,x,1,0,normalize=False)      #Mean

  if not Thiele:
    sig = sqrt(nth_moment(P,x,2,mu,normalize=False)) #Standard deviation
    S   = nth_moment(P,x,3,mu,normalize=False) / sig**3
  else:
    mu_x3 = nth_moment(P,x**3,1,0,normalize=False) #Mean of x cubed
    mu_x2 = nth_moment(P,x**2,1,0,normalize=False) #Mean of x squared
    S     = mu_x3 - 3*mu*mu_x2 + 2*mu**3           #Unnomalized skew

  return S
#------------------------------------------------------------------------------

def testSkewness():
  """
  Test test_add_asym.skewness
  """
  from scipy.stats import skew
  a = np.array([ 0,
                 1, 1,
                 2, 2, 2, 2, 2,
                 3, 3, 3, 3, 3, 3, 3,
                 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                 6, 6, 6, 6, 6, 6, 6, 6, 6, 5,
                 7, 7, 7, 7, 7, 7, 7, 7, 7,
                 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                 9, 9, 9, 9, 9, 9, 9,
                10,10,10,10,10,10,10,10,
                11,11,11,11,11,11,11,
                12,12,12,12,12,12,
                13,13,13,13,13,
                14,14,14,14,14,
                15,15,15,
                16,16,16,16,16,
                17,17,17,
                18,18,18,
                19,19,
                20,20,
                21,21,
                22,22,
                23,23,23,
                24,
                25,
                26,26,
                27,
                28,
                29,29,
                30,
                
                32,
                33,
                34,34,34,
                35,35,
                36,
                37,
                38,
                39,
               
                41,
                42,42,
                43,
                44,
                45,45,
                46,
                
                48,
                49,
                50])
  lena = max(a) - min(a) + 1
  H   = np.histogram(a,bins=lena,range=[0,lena])
  P,x = H[0], H[1][0:lena]
  plt.clf()
  plt.hist(a,bins=lena,range=[0,lena],align='left')
  plt.plot(x,P)
  print('stats.skew:            ', skew(a))
  print('test_add_asym.skewness:', skewness(P,x))
#------------------------------------------------------------------------------

def gauss(x,mu,sigma):
  """
  PDF of a normal distribution.
  """
  return exp(-(x-mu)**2 / (2*sigma**2)) / (sigma*sqrt2pi)
#------------------------------------------------------------------------------

def dimgauss(x,mu,siglo,sighi):
  """
  PDF of a dimidated (asymmetric) Gaussian, i.e. with different stddevs on
  each side of mu.
  """
  assert not isinstance(x,(float,int)), 'If you need x to be a scalar, code it yourself!'
  Glo       = gauss(x,mu,siglo)
  Ghi       = gauss(x,mu,sighi)
  pos       = np.where(x>mu)
  Gdim      = Glo.copy()
  Gdim[pos] = Ghi[pos]
  return Gdim
#------------------------------------------------------------------------------

def lognorm(x,mu,sigma):
  """
  PDF of a lognormal distribution.
  (https://en.wikipedia.org/wiki/Log-normal_distribution)
  The PDF is normalized.
  mu (real):  Location
  sigma (>0): Scale
  """
  assert not isinstance(x,(float,int)), 'If you need x to be a scalar, code it yourself!'
  L      = np.zeros_like(x)
  pos    = np.where(x>0.)
  L[pos] =  exp(-(np.log(x[pos])-mu)**2 / (2*sigma**2)) / (x[pos]*sigma*sqrt2pi)
  return L
#------------------------------------------------------------------------------

def loglogistic(x,a,b):
  """
  PDF of a log-logistic distribution
  (https://en.wikipedia.org/wiki/Log-logistic_distribution).
  The PDF is normalized (but seems to have an extremely long tail).
  a (>0): Scale
  b (>0): Shape
  """
  assert not isinstance(x,(float,int)), 'If you need x to be a scalar, code it yourself!'
  a,b = float(a),float(b)
  xa  = x / a
  L      = np.zeros_like(x)
  pos    = np.where(x>0.)
  L[pos] = b/a * xa[pos]**(b-1) / (1+xa[pos]**b)**2
  return L
#------------------------------------------------------------------------------

def Frechet(x,a,s,m):
  """
  PDF of a Frechet distribution
  (https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution).
  The PDF is normalized (but seems to have an extremely long tail).
  a (>=0):  Shape (this affects skewness)
  s (> 0):  Scale
  m (real): Location
  """
  assert not isinstance(x,(float,int)), 'If you need x to be a scalar, code it yourself!'
  a,s,m  = float(a),float(s),float(m)
  xms    = (x-m) / s
  F      = np.zeros_like(x)
  pos    = np.where(x>m)
  F[pos] = a/s * xms[pos]**(-1-a) * exp(-xms[pos]**(-a))
  return F
#------------------------------------------------------------------------------

def Weibull(x,lam,k):
  """
  PDF of a Weibull distribution
  (https://en.wikipedia.org/wiki/Weibull_distribution).
  The PDF is normalized.
  lam (>0): Scale
  k   (>0): Shape
  """
  assert not isinstance(x,(float,int)), 'If you need x to be a scalar, code it yourself!'
  lam,k  = float(lam),float(k)
  xl     = x / lam
  W      = np.zeros_like(x)
  pos    = np.where(x>0.)
  W[pos] = k/lam * xl[pos]**(k-1) * exp(-xl[pos]**k)
  return W
#------------------------------------------------------------------------------

def skewGauss(x,xi,w,a,A=1.,b=0.):
  """
  Skew normal distribution of the form described at
  http://en.wikipedia.org/wiki/Skew_normal_distribution,
  with two extra added parameters, A and b (see below). Input parameters are x
  (the independent variable) and the following parameters:
    xi: Location parameter (somewhere on the steep side)
    w:  Scale parameter (omega; approximately FWHM)
    a:  Shape parameter (alpha; "asymmetry") For a -> inf, skewness -> 1 (max)
    A:  Overall normalization
    b:  Bias (~ underlying continuum).
  Mean, variance, and skewness can be calculated from the above parameters as:
    d       = a / sqrt(1 + a**2) # (delta)
    mean    = xi + w * d * sqrt(2/pi)
    var     = w**2 * (1 - 2*d**2/pi)
    sigma   = sqrt(var)
    g3      = (4-pi)/2 * (d*sqrt(2/pi))**3 / (1 - 2*d**2/pi)**1.5
  The PDF is normalized (if A = 1 and b = 0).
  """
  A,xi,w,a,b = [float(i) for i in [A,xi,w,a,b]]
  pdf = lambda x: 1/sqrt2pi * exp(-x**2/2)  #Could use scipy.stats.norm.pdf(t)
  cdf = lambda x: (1 + erf(x/sqrt(2))) / 2  #  and scipy.stats.norm.cdf(a*t)
  t   = (x-xi) / w
  return A * 2 / w * pdf(t) * cdf(a*t) + b 
#------------------------------------------------------------------------------

def Gompertz(x,xi,eta,b):
  """
  PDF of a Gompertz distribution
  (https://en.wikipedia.org/wiki/Gompertz_distribution).
  The PDF is normalized (is it though?) and has been generalized with a
  location parameter xi.
  Skewness is always -12*sqrt(6) / pi**2 * zeta(3) ~ 1.3955.
  xi (real): Location
  eta (>0):  Shape
  b   (>0):  Scale
  """
  assert not isinstance(x,(float,int)), 'If you need x to be a scalar, code it yourself!'
  xi,eta,b = float(xi),float(eta),float(b)
  ebx     = exp(b*(x-xi))
# G       = np.zeros_like(x)
# pos     = np.where(x>=-100.)
# G[pos]  = b * eta * ebx[pos] * exp(eta) * exp(-eta*ebx[pos])
  G       = b * eta * ebx      * exp(eta) * exp(-eta*ebx     )
  return G
#------------------------------------------------------------------------------

def Gumbel(x,mu,b):
  """
  PDF of a Gumbel distribution
  (https://en.wikipedia.org/wiki/Gumbel_distribution).
  The PDF is normalized.
  The skewness of this distribution is always ~1.14.
  mu (real): Location
  b (>0):    Scale
  """
  mu,b = float(mu),float(b)
  z    = (x-mu) / b
  return exp(-z-exp(-z)) / b
#------------------------------------------------------------------------------

def plotCI(x,P,p=[0.5,0.1587,0.8413],
           plotit   = True,
           oplot    = False,
           xlim     = None,
           smoothit = False,
           plotsmt  = False,
           verb     = True,
           output   = False,
           outpFWHM = False,
           rectx    = False,    #Rectify non-equidistant x-values
           truncP   = False,    #Set negative P-values to zero
           Lwin     = 25,
           Opol     = 3
           ):
  """
  For a PDF P(x), plot its confidence interval, i.e. the median and two
  percentiles (default 16th and 84th).
  If oplot = True, do not erase old plot.
  If smoothit = True, smooth P before calc'ing moments.
  If plotsmt = True, plot smoothed curve on top of original curve.
  Lwin and Opol are the window length and the polynomial order, respectively,
  of the Savitzky-Golay filter used for smoothing P.

  Output:
  -------
  "Central value", lower error, and upper error for the three cases where the
  central value corresponds to the median, the mean, and the mode,
  respectively. In all cases, errors are calculated as the percentiles given in
  the keyword  'p'. Hence, they are different for the three interpretations of
  the central value.

  Example:
  >>> x  = np.linspace(-5,20,250)
  >>> P1 = test_add_asym.Frechet(x,3,2,0)
  >>> P2 = test_add_asym.lognorm(x,1,.5)
  >>> x0,s1,s2 = test_add_asym.plotCI(x,P1,xlim=[-1,10],oplot=False,verb=False)[0:2]
  >>> print('median,siglo,sighi =', x0,s1,s2)
  median,siglo,sighi = 2.26045049438 0.632298043033 1.33684339632
  >>> test_add_asym.plotCI(x,P2,xlim=[-1,10],oplot=True)
          x0      siglo   sighi
  Median: 2.7187  1.0756  1.7767
  Mean:   3.0796  1.4364  1.4159
  Mode:   2.1285  0.4854  2.3669
  FWHM:   2.61
  """
  if rectx:
    x0,x1,n = x[0],x[-1],len(x)
    xold    = x.copy()
    x       = np.linspace(x0,x1*1.000001,n)
    P       = np.interp(x,xold,P)

  if truncP: P[P<0] = 0
  assert P.min()>=0, "P has negative values. Use truncP=True"
  
  dx  = x[1] - x[0]                                    #\
  xup = x[1:] - x[:-1]                                 # > x should be equidistant
  assert np.less(xup-dx, 1e-6).all(), "x values are not equidistant. Use rectx=True"#/
  P[np.isnan(P)] = 0.                           #Make sure there are no NaNs

  med,lo,hi = wp(x,P,p)                         #Get CI values
  siglo_med = med - lo                          #Lower stddev
  sighi_med = hi  - med                         #Upper stddev
  imed  = (abs(x - med)).argmin()               #\
  ilo   = (abs(x - lo)).argmin()                # > x indices of CI values
  ihi   = (abs(x - hi)).argmin()                #/
  xlohi = x[ilo:ihi+1]                          #Part of x axis in CI
  Plohi = P[ilo:ihi+1]                          #Part of P in CI
  xvec  = np.concatenate([xlohi,xlohi[::-1]])               #Vectors for filled
  yvec  = np.concatenate([np.zeros_like(xlohi),Plohi[::-1]])#  region

  if plotit:
    if not oplot: plt.clf()
    if xlim == None: xlim = [min(x), max(x)]
    plt.xlim(xlim)
    plot = plt.plot(x,P,lw=2)
    col  = plot[0].get_color()                    #Use same color for succeeding plots
    plt.fill(xvec, yvec, col, alpha=0.2)          #CI region
    plt.plot([x[imed],x[imed]],[0,P[imed]],col,label='Median',lw=2)#Median line (solid)

  # Smooth P
  if smoothit:                                  #Smooth P using a Savitzky-
    Pun = P                                     #  Golay filter, but keep
    P   = savgol_filter(P, Lwin, Opol)          #  unsmoothed P

  # Mean
  mu       = nth_moment(P,x,1,0) # simps(x*P,x) / simps(P,x)          #Mean
  siglo_mu = mu - lo                            #\__Calc. errors wrt mu
  sighi_mu = hi - mu                            #/
##stddev   = np.sqrt(nth_moment(P,x,2,mu))
##siglo_mu = mu - stddev                        #\__Calc. errors wrt mu
##sighi_mu = stddev - mu                        #/
  imu      = (abs(x - mu)).argmin()             #x index at mean
  if plotit: plt.plot([x[imu],x[imu]],[0,P[imu]],color=col,ls='--',label='Mean',lw=2)#Mean line (dashed)

  # Mode
  imode      = P.argmax()                       #x index at mode
  mode       = x[imode]
  siglo_mode = mode - lo                        #\__Calc. errors wrt mode
  sighi_mode = hi   - mode                      #/
  if plotit: plt.plot([mode,mode],[0,P[imode]],color=col,ls=':',label='Mode',lw=2)#Mode line (dotted)

  # FWHM
  HM   = P[imode] / 2.                          #Half maximum
  FWlo = (abs(P[:imode]- HM)).argmin()          #Lower P index at HM
  FWhi = (abs(P[imode:] - HM)).argmin() + imode #Upper P index at HM
  FWHM = x[FWhi] - x[FWlo]
  if plotit:
    plt.annotate('', xy=(x[FWlo],HM), xytext=(x[FWhi],HM), #FWHM arrow
      arrowprops=dict(color=col, connectionstyle='arc', arrowstyle='<|-|>'))
    plt.plot([x[FWlo],x[FWhi]], [HM,HM],col,lw=1) #For some reason, the arrow
                                                  #  doesn't go all the way, so
                                                  #  help with a line.

    if not smoothit:
      plt.plot(x,P,'k')
    else:
      plt.plot(x,Pun,'k')
      if plotsmt: plt.plot(x,P,'k')

    plt.legend(loc=1,fontsize='medium',frameon=False,handlelength=3.5)

  if verb:
    if med>1e3: print("Change the f's below to g's for nicer formatting")
    print('        x0      siglo   sighi')
    print('Median: {:.4f}  {:.4f}  {:.4f}'.format(med, siglo_med, sighi_med))
    print('Mean:   {:.4f}  {:.4f}  {:.4f}'.format(mu,  siglo_mu,  sighi_mu))
    print('Mode:   {:.4f}  {:.4f}  {:.4f}'.format(mode,siglo_mode,sighi_mode))
    print('FWHM:   {:.4g}'.format(FWHM))

  if output:
    return med, siglo_med, sighi_med, \
           mu,  siglo_mu,  sighi_mu,  \
           mode,siglo_mode,sighi_mode
  elif outpFWHM:
    return FWHM
#------------------------------------------------------------------------------

def testasym(x,P1,P2,Lwin=51,Opol=3,smoothit=True,xlim=[-1,10],verb=True,plotit=True):
  """
  Given the probability distributions P1 and P2 defined on x,
    1. Using plotCI(), determine their confidence intervals CI1 and CI2
      {x0,siglo,sighi}, for the three interpretations of x0: median, mean, mode;
    2. Monte Carlo sample the sum P(1,2), and determine CI3 likewise;
    3. Use CI1 and CI2 to determine the combined CI3 with add_asym();
    4. Compare the two CI3's individually for x0, siglo, and sighi, for all
       three x0's (i.e. med, mu, and mode);
    5. Return the obtained relative ratios      ### , along with the skewness of P1 and P2.
  Usage:
    >>> x  = np.linspace(-5,20,2501)
    >>> P1 = test_add_asym.lognorm(x,1,.5)
    >>> P2 = test_add_asym.Frechet(x,3,2,0)
    >>> test_add_asym.testasym(x,P1,P2)
  """
  from scipy.stats import skew

  # 1. Get CI's from plotCI
  med1,siglo_med1,sighi_med1,mu1,siglo_mu1,sighi_mu1,mode1,siglo_mode1,sighi_mode1 \
    = plotCI(x,P1,xlim=xlim,plotit=plotit,output=True,verb=False)
  med2,siglo_med2,sighi_med2,mu2,siglo_mu2,sighi_mu2,mode2,siglo_mode2,sighi_mode2 \
    = plotCI(x,P2,xlim=xlim,plotit=plotit,output=True,verb=False,oplot=True)

  # 2.1 Get CI1 and CI2 from Monte Carlo sampling
  R1        = np.random.choice(x,10000000,p=P1/sum(P1))
  R2        = np.random.choice(x,10000000,p=P2/sum(P2))
  P1MC,dum  = np.histogram(R1,bins=len(x),range=[min(x),max(x)],density=True)
  P2MC,dum  = np.histogram(R2,bins=len(x),range=[min(x),max(x)],density=True)

  # 2.2 Add CI's and get CI3 from plotCI
  MCsum,dum = np.histogram(R1+R2,bins=len(x),range=[min(x),max(x)],density=True)
  if plotit: plt.ylim([0,.4])
  med3,siglo_med3,sighi_med3,mu3,siglo_mu3,sighi_mu3,mode3,siglo_mode3,sighi_mode3 \
    = plotCI(x,MCsum,xlim=xlim,plotit=plotit,oplot=True,smoothit=smoothit,plotsmt=True,output=True,verb=False)
  CI3_med  = np.array([med3, siglo_med3, sighi_med3])
  CI3_mu   = np.array([mu3,  siglo_mu3,  sighi_mu3])
  CI3_mode = np.array([mode3,siglo_mode3,sighi_mode3])
  if plotit: plt.plot(x,P1MC,alpha=.5)
  if plotit: plt.plot(x,P2MC,alpha=.5)

  # 3.1 Define central values and CIs for add_asym()
  x0_med  = [med1,med2]
  s1_med  = [siglo_med1,siglo_med2]
  s2_med  = [sighi_med1,sighi_med2]
  x0_mu   = [mu1,mu2]
  s1_mu   = [siglo_mu1,siglo_mu2]
  s2_mu   = [sighi_mu1,sighi_mu2]
  x0_mode = [mode1,mode2]
  s1_mode = [siglo_mode1,siglo_mode2]
  s2_mode = [sighi_mode1,sighi_mode2]

  # 3.2 Get CI3's from add_asym
  add0_med  = add_asym(x0_med, s1_med, s2_med, order=0,ohwell=True)
  add1_med  = add_asym(x0_med, s1_med, s2_med, order=1)
  add2_med  = add_asym(x0_med, s1_med, s2_med, order=2)
  add0_mu   = add_asym(x0_mu,  s1_mu,  s2_mu,  order=0,ohwell=True)
  add1_mu   = add_asym(x0_mu,  s1_mu,  s2_mu,  order=1)
  add2_mu   = add_asym(x0_mu,  s1_mu,  s2_mu,  order=2)
  add0_mode = add_asym(x0_mode,s1_mode,s2_mode,order=0,ohwell=True)
  add1_mode = add_asym(x0_mode,s1_mode,s2_mode,order=1)
  add2_mode = add_asym(x0_mode,s1_mode,s2_mode,order=2)

  # Print output for x0 = median
  if verb:
  # print('Int(P1+P2)                 ', simps(MCsum,x))
    print('Exact convolution:  {:5.2f} -{:.2f} +{:.2f}'.format(*CI3_med))
    print('WRONG oh so wrong!: {:5.2f} -{:.2f} +{:.2f}'.format(*add0_med))
    print('Linear transf.:     {:5.2f} -{:.2f} +{:.2f}'.format(*add1_med))
    print('Quadratic transf.:  {:5.2f} -{:.2f} +{:.2f}'.format(*add2_med))

  # Calculate offset
  #For each order [0,1,2], calculate the relative ratios between the set
  # [med,siglo,sighi] as returned by add_asym() and the "true" MC method.
  rat0_med  = (add0_med  - CI3_med ) / CI3_med
  rat1_med  = (add1_med  - CI3_med ) / CI3_med
  rat2_med  = (add2_med  - CI3_med ) / CI3_med
  rat0_mu   = (add0_mu   - CI3_mu  ) / CI3_mu
  rat1_mu   = (add1_mu   - CI3_mu  ) / CI3_mu
  rat2_mu   = (add2_mu   - CI3_mu  ) / CI3_mu
  rat0_mode = (add0_mode - CI3_mode) / CI3_mode
  rat1_mode = (add1_mode - CI3_mode) / CI3_mode
  rat2_mode = (add2_mode - CI3_mode) / CI3_mode

# print('If estimator is {med,16,84}, then')
# print(' - order 0 is off by {:.1f},{:.1f},{:.1f} percent'.format(*(100*rat0_med)))
# print(' - order 1 is off by {:.1f},{:.1f},{:.1f} percent'.format(*(100*rat1_med)))
# print(' - order 2 is off by {:.1f},{:.1f},{:.1f} percent'.format(*(100*rat1_med)))

# print('If estimator is {mu,16,84}, then')
# print(' - order 0 is off by {:.1f},{:.1f},{:.1f} percent'.format(*(100*rat0_mu)))
# print(' - order 1 is off by {:.1f},{:.1f},{:.1f} percent'.format(*(100*rat1_mu)))
# print(' - order 2 is off by {:.1f},{:.1f},{:.1f} percent'.format(*(100*rat1_mu)))

# print('If estimator is {mode,16,84}, then')
# print(' - order 0 is off by {:.1f},{:.1f},{:.1f} percent'.format(*(100*rat0_mode)))
# print(' - order 1 is off by {:.1f},{:.1f},{:.1f} percent'.format(*(100*rat1_mode)))
# print(' - order 2 is off by {:.1f},{:.1f},{:.1f} percent'.format(*(100*rat1_mode)))

  # Return [order 0,1,2] X [med,mu,mode] X [x0,siglo,sighi] = 27 values:
  #           order 0    order 1    order 2
  # median  [x0,s1,s2] [x0,s1,s2] [x0,s1,s2]
  # mu      [x0,s1,s2] [x0,s1,s2] [x0,s1,s2]
  # mode    [x0,s1,s2] [x0,s1,s2] [x0,s1,s2]
  return rat0_med,  rat1_med,  rat2_med, \
         rat0_mu,   rat1_mu,   rat2_mu,  \
         rat0_mode, rat1_mode, rat2_mode
#------------------------------------------------------------------------------

def testemall(nsample=1000,fname='add_asym.dat'):
  """
  Using several different probability distributions of various skewness as
  input  to testasym(), test the function add_asym(). For each pair of PDFs,
  testasym() outputs the ratio between add_asym()'s predicted CI
  (i.e. {x0,siglo,sighi}) and the "real" CI (from Monte Carlo sampling).
  This ratio is given for order 0, 1, and 2. Furthermore, all 9 values are
  calculated for x0 corresponding to the median, mean, and mode (saved in
  different files for each x0 type).
  """
  x        = np.linspace(-20,200,220000)
  res_med  = []
  res_mu   = []
  res_mode = []

  for i in range(nsample):
    print(i)
    P1,n1 = getRandomDistr(x,allowInverted=False) # gauss(x,0.,1.),-1 #
    P2,n2 = getRandomDistr(x,allowInverted=False)
    skew1 = skewness(P1,x)
    skew2 = skewness(P2,x)
    rat0_med,rat1_med,rat2_med,rat0_mu,rat1_mu,rat2_mu,rat0_mode,rat1_mode,rat2_mode \
      = testasym(x,P1,P2,verb=False,plotit=False)
    res_med.append( [n1,n2,skew1,skew2,rat0_med [0],rat0_med [1],rat0_med [2],
                                       rat1_med [0],rat1_med [1],rat1_med [2],
                                       rat2_med [0],rat2_med [1],rat2_med [2]])
    res_mu.append(  [n1,n2,skew1,skew2,rat0_mu  [0],rat0_mu  [1],rat0_mu  [2],
                                       rat1_mu  [0],rat1_mu  [1],rat1_mu  [2],
                                       rat2_mu  [0],rat2_mu  [1],rat2_mu  [2]])
    res_mode.append([n1,n2,skew1,skew2,rat0_mode[0],rat0_mode[1],rat0_mode[2],
                                       rat1_mode[0],rat1_mode[1],rat1_mode[2],
                                       rat2_mode[0],rat2_mode[1],rat2_mode[2]])

  np.savetxt('add_asym_med.dat',np.array(res_med),
    fmt='%4d %2d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f',
    header='n1 n2   skew1   skew2    med0     lo0     hi0    med1     lo1     hi1    med2     lo2     hi2')
  np.savetxt('add_asym_mu.dat',np.array(res_mu),
    fmt='%4d %2d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f',
    header='n1 n2   skew1   skew2    med0     lo0     hi0    med1     lo1     hi1    med2     lo2     hi2')
  np.savetxt('add_asym_mode.dat',np.array(res_mode),
    fmt='%4d %2d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f',
    header='n1 n2   skew1   skew2    med0     lo0     hi0    med1     lo1     hi1    med2     lo2     hi2')
#------------------------------------------------------------------------------

def getRandomDistr(x,allowInverted=True):
  """
  Return a random PDF defined on x.
  Most of these PDFs have a positive skewness. The keyword allowInverted
  gives the possibility of flipping the selected PDF.
  """
  #Lists of possible PDFs
  PDFs    = [
             lognorm,
             loglogistic,
             Frechet,
             Weibull,
            #skewGauss,
            #Gompertz,    # Gave up on this one; difficult to normalize (min skew -1.14?)
            #Gumbel       # Found out thin one had constant skewness 1.14
             ]
  # List of ranges of arguments for these PDFs. The ranges have been chosen
  # to be able to give functions of high skewness while still being able to
  # be properly normalized given the dx size for x = linspace(-200,200,400000).
  # The location parameters are a bit random...
  allargs = [
             [[0],    [.1,1.5]       ],       #1. Lognorm:     mu,sigma
             [[5],    [2,30]         ],       #2. Loglogistic: a,b
             [[3,11], [3,9], [0]     ],       #3. Frechet:     a,s,m
             [[5],    [1.5,3]        ],       #4. Weibull:     lam, k
            #[[5],    [5,17], [3,30] ],       #5. skewGauss:   xi,w,a
            #[[20],   [.01,1], [.1,9]],       #6. Gompertz:    xi,eta,b
            #[[5],    [1,17]         ]        #7. Gumbel:      mu,b
             ]
  n   = np.random.randint(len(PDFs))          #Function number (0-6)
  PDF = PDFs[n]                               #Random PDF

  i   = 0                                     #Counter for failed PDFs
  while True:
    i += 1
    assert i<100, 'Fuck this shit, I quit.'   #Took too long to find a well-behaved PDF
    args = []                                 #Create current arguments
    for a in allargs[n]:
      if len(a) == 1:
        arg = a[0]
      else:
        arg = np.random.uniform(*a)
      args.append(arg)
    P = PDF(x,*args)
    if abs(simps(P,x) - 1) < 1e-3: break      #Accept only if normalizable

# print('function   =', PDF.__name__)
# print('arguments  =', args)

  if allowInverted:
    if np.random.choice([0,1]): P = P[::-1]   #Flip P with probability 50%
# print(n,args)
  return P,n
#-----------------------------------------------------------------------------

def plotTests(order,nn='all',
              data  = 'med',
              x0    = 'med',
              cmap  = 'viridis',
              crange= [-.25,.05],   #Color range (in fraction; not percent)
              output= False,        #Output 1D result
              pdf   = False):
  """
  Visualize how accurate a given method of adding asymmetric uncertainties is.

  The commented out part plots fractional difference between the result of
  adding two skewed functions by "some method" and the true result obtained
  through Monte Carlo sampling, as a function of the average skewnnes of the
  two functions.

  Here, "some method" refers either to the standard, but wrong, quadrature
  addition (order 0), or to the add_asym() method (order 1 or 2).

  The part below plots the same, but as a function of the skewness of both
  functions, i.e. a 3D plot.

  Keywords:
  ---------
    data:
      'med' - plot fractional difference for the median (or mode or mean,
              if 'x0' is set to this)
      'lo'  - plot fractional difference for the lower std dev
      'hi'  - plot fractional difference for the upper std dev
    x0:
      Interpretation of "the central value". Can take the values 'med', 'mode',
      and 'mu', for median, mode and mean.
  Examples:
  --------
  >>> test_add_asym.plotTests(0,data='med')     # plot bad method
  >>> test_add_asym.plotTests(1,data='med')     # plot good method
  >>> test_add_asym.plotTests(1,data='med',x0='mode')#plot good method for the mode

  Two figures:
  ------------
  To plot two figures (say order 0 and order 1 for medians) next to each other,
  plot them separately, outcommenting the part of the code that renders the
  colorbar, and merge them in Keynote. Elegant as fuck.
  """
  import itertools

  fname = './add_asym/add_asym_'+x0+'_normdDiffs.dat'
  n1,n2,skew1,skew2, med0,lo0,hi0,med1,lo1,hi1,med2,lo2,hi2 = \
    np.loadtxt(fname,unpack=True)

  n       = len(n1)                             #Total number of data points
  PDFs    = ['lognorm','loglogistic','Frechet','Weibull']
  colors  = ['r','g','b','y']                   #Colors for different PDFs
  markers = ['o','2','+','x']                   #Markers for different PDFs
  title   = ['"Usual" quadrature adding propagation',
             'Piecewise linear propagation',
             'Quadratic propagation']

  aveskew = (skew1+skew2) / 2                   #Plot as function of average skew
  exec("med = med" + str(order))                #Use keyword 'order' to map
  exec("lo  = lo"  + str(order))                #  the right order data to the
  exec("hi  = hi"  + str(order))                #  variables med, lo, and hi

  # Indices for combinations of the four functions
  i00,i01,i02,i03,i10,i11,i12,i13,i20,i21,i22,i23,i30,i31,i32,i33 =  \
    np.where((n1==0) & (n2==0))[0], np.where((n1==0) & (n2==1))[0], \
    np.where((n1==0) & (n2==2))[0], np.where((n1==0) & (n2==3))[0], \
    np.where((n1==1) & (n2==0))[0], np.where((n1==1) & (n2==1))[0], \
    np.where((n1==1) & (n2==2))[0], np.where((n1==1) & (n2==3))[0], \
    np.where((n1==2) & (n2==0))[0], np.where((n1==2) & (n2==1))[0], \
    np.where((n1==2) & (n2==2))[0], np.where((n1==2) & (n2==3))[0], \
    np.where((n1==3) & (n2==0))[0], np.where((n1==3) & (n2==1))[0], \
    np.where((n1==3) & (n2==2))[0], np.where((n1==3) & (n2==3))[0]

  # Lengths of index array
  l00,l01,l02,l03,l10,l11,l12,l13,l20,l21,l22,l23,l30,l31,l32,l33 = \
    len(i00),len(i01),len(i02),len(i03),len(i10),len(i11),len(i12),len(i13), \
    len(i20),len(i21),len(i22),len(i23),len(i30),len(i31),len(i32),len(i33)
  ni = sum([l00,l01,l02,l03,l10,l11,l12,l13,l20,l21,l22,l23,l30,l31,l32,l33])
  assert (n == ni), 'Total number of data points is ' + str(n) + ', but only ' + str(ni) + ' were caught'

  # Pick out values using the above indices
  med00,lo00,hi00 = med[i00],lo[i00],hi[i00]
  med01,lo01,hi01 = med[i01],lo[i01],hi[i01]
  med02,lo02,hi02 = med[i02],lo[i02],hi[i02]
  med03,lo03,hi03 = med[i03],lo[i03],hi[i03]
  med10,lo10,hi10 = med[i10],lo[i10],hi[i10]
  med11,lo11,hi11 = med[i11],lo[i11],hi[i11]
  med12,lo12,hi12 = med[i12],lo[i12],hi[i12]
  med13,lo13,hi13 = med[i13],lo[i13],hi[i13]
  med20,lo20,hi20 = med[i20],lo[i20],hi[i20]
  med21,lo21,hi21 = med[i21],lo[i21],hi[i21]
  med22,lo22,hi22 = med[i22],lo[i22],hi[i22]
  med23,lo23,hi23 = med[i23],lo[i23],hi[i23]
  med30,lo30,hi30 = med[i30],lo[i30],hi[i30]
  med31,lo31,hi31 = med[i31],lo[i31],hi[i31]
  med32,lo32,hi32 = med[i32],lo[i32],hi[i32]
  med33,lo33,hi33 = med[i33],lo[i33],hi[i33]

  # Select data to show
  exec("medlohi = " + data)

  # Plot offset as a function of average skewness
# plt.clf()
# plt.xlim([.01,10])
# plt.ylim([-.5,.1])
# plt.xlabel('Average skewness')

# a = .5
# if (nn=='all') or (nn=='00'): plt.scatter(aveskew[i00],medlohi[i00],s=15,alpha=a,color=colors[0],marker=markers[0],edgecolor='')
# if (nn=='all') or (nn=='01'): plt.scatter(aveskew[i01],medlohi[i01],s=20,alpha=a,color=colors[0],marker=markers[1])
# if (nn=='all') or (nn=='02'): plt.scatter(aveskew[i02],medlohi[i02],s=20,alpha=a,color=colors[0],marker=markers[2])
# if (nn=='all') or (nn=='03'): plt.scatter(aveskew[i03],medlohi[i03],s=20,alpha=a,color=colors[0],marker=markers[3])
# if (nn=='all') or (nn=='10'): plt.scatter(aveskew[i10],medlohi[i10],s=15,alpha=a,color=colors[1],marker=markers[0],edgecolor='')
# if (nn=='all') or (nn=='11'): plt.scatter(aveskew[i11],medlohi[i11],s=20,alpha=a,color=colors[1],marker=markers[1])
# if (nn=='all') or (nn=='12'): plt.scatter(aveskew[i12],medlohi[i12],s=20,alpha=a,color=colors[1],marker=markers[2])
# if (nn=='all') or (nn=='13'): plt.scatter(aveskew[i13],medlohi[i13],s=20,alpha=a,color=colors[1],marker=markers[3])
# if (nn=='all') or (nn=='20'): plt.scatter(aveskew[i20],medlohi[i20],s=15,alpha=a,color=colors[2],marker=markers[0],edgecolor='')
# if (nn=='all') or (nn=='21'): plt.scatter(aveskew[i21],medlohi[i21],s=20,alpha=a,color=colors[2],marker=markers[1])
# if (nn=='all') or (nn=='22'): plt.scatter(aveskew[i22],medlohi[i22],s=20,alpha=a,color=colors[2],marker=markers[2])
# if (nn=='all') or (nn=='23'): plt.scatter(aveskew[i23],medlohi[i23],s=20,alpha=a,color=colors[2],marker=markers[3])
# if (nn=='all') or (nn=='30'): plt.scatter(aveskew[i30],medlohi[i30],s=15,alpha=a,color=colors[3],marker=markers[0],edgecolor='')
# if (nn=='all') or (nn=='31'): plt.scatter(aveskew[i31],medlohi[i31],s=20,alpha=a,color=colors[3],marker=markers[1])
# if (nn=='all') or (nn=='32'): plt.scatter(aveskew[i32],medlohi[i32],s=20,alpha=a,color=colors[3],marker=markers[2])
# if (nn=='all') or (nn=='33'): plt.scatter(aveskew[i33],medlohi[i33],s=20,alpha=a,color=colors[3],marker=markers[3])
#
# for p,c,m in zip(PDFs,colors,markers): plt.scatter([-1],[0],s=25,label=p,color=c,marker=m)
# plt.legend(loc=3,fontsize='large',frameon=False,handlelength=3.5,scatterpoints=1,handletextpad=-1)

  # Create matrix
  min1 = 0.                                     #\
  min2 = 0.                                     # \_ min/max for skewness axes
  max1 = 10.                                    # /
  max2 = 10.                                    #/
  nrow = 20                                     #\_ define grid to sample data in
  ncol = 20                                     #/
  res  = np.array([[0. for i in range(ncol)] for j in range(nrow)])#Initialize grid
  ndat = np.zeros_like(res)#, np.int32)         #Array for # of data points/bin
  for i in range(n):
    i1 = int(.99999 * skew1[i] / max1 * ncol)
    i2 = int(.99999 * skew2[i] / max2 * nrow)
    res [i1,i2] += medlohi[i]
    ndat[i1,i2] += 1
  res = res / ndat                              #Take average
  f   = 100                                     #1 for fractional, 100 for %
  res = f * res
  pd.options.display.float_format = ' {:,.2f}'.format
# print(pd.DataFrame(res))

  # Plot offset = offset(skew1,skew2)
  xlo,xhi = min1,max1         #Data coordinates
  ylo,yhi = min2,max2         #

  plt.close()                 #This is just to close previous rendition
  fig = plt.figure(figsize=(6.9,6), facecolor='w')
  plt.rcParams.update({'font.size': 15})
  ax  = fig.add_subplot(1,1,1)
  ax.set_xlabel('Skewness 1')
  ax.set_ylabel('Skewness 2')
  ax.set_title(title[order])
  im  = ax.imshow(res,
    cmap=cmap,              #Most colormaps have reversed versions appended with '_r'
    extent=[xlo,xhi,yhi,ylo], #Note order of ylo and yhi! This makes y increase from bottom to top
  # aspect='auto',            #Force res to be square, even if nrow != ncol
    vmin=crange[0]*f,vmax=crange[1]*f,         #Min/max values for the color range
    interpolation='nearest')  #Omit this to smooth pixels
  plt.gca().invert_yaxis()    #Flip y axis so it increases from bottom to top

  # Plot colorbar
  cbar = fig.colorbar(im, ax=ax, #Not sure what 'ax=ax' does
    orientation='vertical',      #This is default
    ticks=np.linspace(-f,f,41),  #Values drawn only go to vmax, so it doesn't matter if range is longer
    fraction=.05)                #Shitty way to manipulate length of colorbar
  cbar.ax.set_ylabel('% difference between result and "true" result', rotation=90)
  plt.draw()                  #Isn't necessary, I think

  # Make pdf
  if pdf != False: plt.savefig(pdf, bbox_inches='tight')

  # Extract values along diagonal of res (at double the resolution)
  if output:
    assert (nrow==ncol) & (min1==min2) & (max1==max2), '1D result only applicable if {nrow,min1,max1} = {ncol,min2,max2}.'
    off = np.zeros(2*nrow)
    for i in range(nrow):
      for j in range(nrow):
        if i==j:   off[2*i+1] =  res[i,j]
        if i-j==1: off[2*i  ] = (res[i,j]+res[j,i]) / 2
    S = np.linspace(min1,max1,2*nrow)
  # plt.clf()
  # plt.plot(S,off)
  # plt.scatter(S,off)
    return S,off
#------------------------------------------------------------------------------

def plot1Doffset(data):
  if data == 'old':
  # Old results before correct (?) sigmas:
    skew      = np.array([0.,0.52631579,1.05263158,1.57894737,2.10526316,2.63157895,3.15789474,3.68421053,4.21052632,4.73684211,5.26315789,5.78947368,6.31578947,6.84210526,7.36842105,7.89473684,8.42105263,8.94736842,9.47368421,10.])
    med_x00   = np.array([0.,-0.86169714,-1.26030216,-1.72544643,-2.12040055,-2.43584071,-3.28622449,-4.70357143,-5.25258621,-6.97692308,-5.66363636,-11.12,-7.26357143,-7.95,-10.58333333,-12.95,-21.95,-22.78,-25.315,-28.14074074])
    med_lo0   = np.array([0.,-1.43680263,-3.36404151,-7.47589286,-9.798955,-12.4420354,-13.22442466,-15.63928571,-15.8830819,-18.69230769,-16.86464646,-21.38,-16.43642857,-16.34,-21.45833333,-26.55,-34.4,-35.32,-38.06,-40.97037037])
    med_hi0   = np.array([0.,0.48785748,0.10733578,-0.98035714,-1.66715622,-2.77831858,-3.45806557,-4.87142857,-5.50215517,-6.7,-6.95151515,-8.86,-7.84285714,-8.01,-10.30833333,-12.15,-15.5,-16.58,-17.99,-21.23333333])
    mode_x00  = np.array([0.,-2.97310111,-3.88051593,-4.44481132,-5.36807451,-5.5984127,-7.23787202,-11.72173913,-14.88125,-14.58823529,-14.229375,-19.8,-17.93504274,-17.82,-29.19166667,-38.3,-57.5,-62.2,-62.585,-66.85714286])
    mode_lo0  = np.array([0.,-1.45432995,-3.38549505,-7.73301887,-9.50170349,-12.24722222,-13.16227679,-15.21086957,-17.06208333,-15.84705882,-16.22229167,-18.63,-17.73076923,-17.52,-23.425,-29.1,-34.4,-35.6,-37.81,-41.37857143])
    mode_hi0  = np.array([0.,0.50551524,0.09284259,-1.21273585,-1.57882391,-2.81666667,-3.28809524,-5.04130435,-5.83666667,-6.00588235,-6.39854167,-8.,-8.00982906,-7.42,-11.10833333,-13.9,-15.8,-16.7,-17.92,-21.52857143])
    mu_x00    = np.array([0.,0.00096386,0.00088133,0.,-0.00047917,0.00046729,0.00518258,-0.0027027,0.0192029,0.01578947,-0.00456731,0.02142857,0.0025,0.025,-0.02666667,0.,0.045,0.1,0.00833333,-0.01111111])
    mu_lo0    = np.array([0.,-1.44838554,-3.31487165,-7.69269406,-9.26048215,-11.97009346,-13.49813202,-15.58918919,-16.69157609,-15.71578947,-17.49014423,-18.86428571,-17.99166667,-21.3625,-22.99666667,-23.5,-30.02,-35.5,-36.525,-40.6])
    mu_hi0    = np.array([0.,0.47108434,0.14860877,-1.03607306,-1.53098906,-2.8135514,-3.4838764,-4.98918919,-5.60706522,-6.00526316,-6.78894231,-8.22142857,-7.65416667,-9.95,-10.66666667,-11.85,-13.46,-16.7,-17.225,-20.77777778])
    med_x01   = np.array([0.,-0.20300047,-0.26798649,-0.28392857,-0.32108179,-0.29646018,-0.37516283,-0.43571429,-0.4643319,-0.42307692,-0.44040404,-0.3,-0.665,-0.84,-0.50833333,0.45,1.45,1.72,2.28,4.02592593])
    med_lo1   = np.array([0.,-0.22090952,-1.11092384,-2.52232143,-3.37493902,-4.3,-4.63148068,-5.44285714,-5.57295259,-6.33076923,-5.82777778,-6.8,-5.96928571,-6.12,-7.44166667,-7.35,-8.75,-8.66,-9.355,-8.38888889])
    med_hi1   = np.array([0.,-0.52550398,-1.70607669,-4.18883929,-5.54264403,-7.30840708,-8.02077725,-9.91785714,-10.47155172,-12.24615385,-12.09444444,-14.58,-12.45571429,-12.46,-15.70833333,-19.1,-23.175,-24.3,-25.4475,-28.97037037])
    mode_x01  = np.array([0.,-2.2670537,-2.79127317,-2.97169811,-3.51320446,-3.43690476,-4.19077381,-6.32173913,-7.30875,-7.6,-6.6,-8.83,-7.55641026,-9.4,-10.24166667,-11.5,-12.05,-13.5,1.78,32.13571429])
    mode_lo1  = np.array([0.,-0.20977262,-1.0967481,-2.61981132,-3.27890947,-4.20357143,-4.62224702,-5.12173913,-5.74958333,-5.73529412,-5.82,-6.38,-6.17051282,-6.68,-7.48333333,-8.05,-8.85,-8.6,-9.21,-8.33214286])
    mode_hi1  = np.array([0.,-0.52917271,-1.75631741,-4.56273585,-5.3494282,-7.33095238,-7.80498512,-10.11304348,-11.16208333,-10.75882353,-11.16270833,-13.19,-12.91196581,-11.82,-17.18333333,-21.05,-23.4,-24.5,-25.51,-29.26785714])
    mu_x01    = np.array([0.,0.63484337,0.98731809,1.43378995,1.54718573,1.9635514,2.43636938,3.78918919,5.34338768,3.93157895,5.32067308,7.06428571,6.71416667,9.1625,8.57,7.5,12.205,16.3,15.40833333,17.71944444])
    mu_lo1    = np.array([0.,-0.22284337,-1.0818583,-2.60319635,-3.23386409,-4.13457944,-4.65660112,-5.31621622,-5.7263587,-5.70526316,-6.06995192,-6.17857143,-6.43916667,-6.7375,-7.22666667,-7.4,-8.08,-8.7,-9.09166667,-8.53611111])
    mu_hi1    = np.array([0.,-0.55339759,-1.63628469,-4.32876712,-5.1826398,-7.24065421,-8.1715309,-10.15675676,-10.74438406,-10.81578947,-11.88557692,-13.60714286,-12.40166667,-15.6875,-16.79333333,-18.2,-20.74,-24.5,-24.76666667,-28.50555556])
    med_x02   = np.array([0.,-4.23347398e-02,  -1.33780472e-02, 8.75000000e-02,   1.53650126e-01,   2.83185841e-01, 4.03039514e-01,   7.46428571e-01,   8.53987069e-01, 1.42307692e+00,   1.01111111e+00,   2.78000000e+00, 1.17571429e+00,   1.18000000e+00,   2.44166667e+00, 4.35000000e+00,   8.25000000e+00,   8.90000000e+00, 1.03175000e+01,   1.35259259e+01])
    med_lo2   = np.array([0.,-0.02372246,  -0.46411251,  -1.00446429, -1.0810103 ,  -1.16150442,  -0.98376031,  -0.76785714, -0.56616379,   0.10769231,  -0.25151515,   1.76      , -0.32928571,  -0.25      ,   1.36666667,   3.95      , 8.575     ,   9.44      ,  11.33      ,  15.66666667])
    med_hi2   = np.array([0.,-0.35738397,  -1.15495767,  -3.20803571, -4.11523654,  -5.57256637,  -6.0542119 ,  -7.64642857, -8.06551724,  -9.36923077,  -9.53585859, -11.12      , -10.05571429, -10.02      , -12.39166667, -14.75      , -17.425     , -18.42      , -19.1375    , -22.24074074])
    mode_x02  = np.array([0.,-2.08780842,  -2.51663229,  -2.58962264, -3.03180312,  -2.85515873,  -3.37440476,  -4.84130435, -5.21083333,  -5.64705882,  -4.469375  ,  -5.71      , -4.61367521,  -7.02      ,  -4.775     ,  -3.65      , 1.2       ,   0.8       ,  20.56      ,  61.40357143])
    mode_lo2  = np.array([0.,-4.11223996e-03,  -4.41505542e-01, -1.06603774e+00,  -1.06727096e+00,  -1.16150794e+00, -1.00944940e+00,  -5.95652174e-01,  -2.34166667e-01, -5.00000000e-01,  -4.37083333e-01,   4.30000000e-01, 2.11538462e-01,  -4.20000000e-01,   2.41666667e+00, 5.55000000e+00,   8.20000000e+00,   9.90000000e+00, 1.11750000e+01,   1.62000000e+01])
    mode_hi2  = np.array([0.,-0.35462022,  -1.19733212,  -3.54764151, -3.97400693,  -5.63412698,  -5.85208333,  -7.84347826, -8.52208333,  -8.4       ,  -8.72916667, -10.36      , -10.24358974,  -9.22      , -13.35833333, -16.15      , -17.75      , -18.5       , -19.235     , -22.48214286])
    mu_x02    = np.array([0., 0.79359036,   1.23312109,   1.80730594, 1.9493248 ,   2.48598131,   3.08648174,   4.81891892, 6.82681159,   5.03157895,   6.80288462,   9.06428571, 8.63083333,  11.8       ,  10.97666667,   9.65      , 15.69      ,  21.        ,  19.84166667,  22.95833333])
    mu_lo2    = np.array([0.,-0.02144578,  -0.43414801,  -1.02328767, -1.11286998,  -1.18037383,  -0.9420014 ,  -0.66486486, -0.24067029,  -0.72631579,  -0.02716346,   0.76428571, 0.20583333,   2.        ,   2.25      ,   2.4       , 5.855     ,   9.6       ,  10.34166667,  15.06111111])
    mu_hi2    = np.array([0.,-0.38009639,  -1.09251742,  -3.30456621, -3.87103767,  -5.59018692,  -6.15572331,  -7.83783784, -8.1798913 ,  -8.46315789,  -9.18125   , -10.70714286, -9.68583333, -12.2875    , -13.02666667, -14.35      , -15.74      , -18.6       , -18.525     , -21.82777778])
  elif data == 'new':
    skew      = np.array([0., 0.25641026, 0.51282051, 0.76923077, 1.02564103, 1.28205128, 1.53846154, 1.79487179, 2.05128205, 2.30769231, 2.56410256, 2.82051282, 3.07692308, 3.33333333, 3.58974359, 3.84615385, 4.1025641, 4.35897436, 4.61538462, 4.87179487, 5.12820513, 5.38461538, 5.64102564, 5.8974359, 6.15384615, 6.41025641, 6.66666667, 6.92307692, 7.17948718, 7.43589744, 7.69230769, 7.94871795, 8.20512821, 8.46153846, 8.71794872, 8.97435897, 9.23076923, 9.48717949, 9.74358974, 10.        ])
    med_x00   = np.array([0., -0.4286675, -0.84455842, -1.54045151, -1.64769779, -1.86595745, -1.74069032, -1.56298269, -1.82538781, -2.04688427, -2.58861558, -3.24271523, -3.76091479, -3.9302521, -4.74250936, -5.34571429, -5.11356606, -5.23255814, -5.94946581, -6.18387097, -7.21549539, -6.335, -7.40132488, -7.75416667, -9.55678261, -9.83125, -10.34715909, -10.38947368, -12.9102381, -13.55833333, -17.66428571, -21.15, -21.84107143, -19.9, -21.94047619, -23.44285714, -25.60199275, -28.65588235, -27.92923077, -28.17058824])
    med_lo0   = np.array([0., -0.89465124, -1.35852142, -2.3739461, -3.16047924, -4.18297872, -6.59808247, -9.96697736, -10.69392677, -11.85054402, -12.6562128, -13.47483444, -14.22923872, -14.28235294, -15.09422987, -16.04142857, -15.58992488, -16.11860465, -16.9724359, -18.04193548, -18.3843318, -17.1275, -18.27229263, -18.39166667, -20.51921739, -20.8375, -20.78267045, -18.16315789, -22.49547619, -27.40833333, -30.76714286, -33.55, -34.325, -33.1, -34.73571429, -36.07142857, -38.31847826, -41.49411765, -40.74944056, -41.02156863])
    med_hi0   = np.array([0., 0.06715671, 0.58777205, 0.92195347, 0.2007048, -0.31884498, -0.85154135, -1.76977364, -1.91427633, -2.57903066, -2.97107118, -3.68046358, -4.09591479, -4.46302521, -4.83931414, -5.58714286, -5.55048608, -6.04883721, -6.39294872, -7.08709677, -7.23456221, -6.8275, -7.515553, -7.95, -8.9746087, -9.31875, -9.46704545, -8.84210526, -10.42333333, -12.38333333, -14.04142857, -15.3, -15.61071429, -15.44, -16.3, -17.15714286, -18.28487319, -21.55392157, -21.05769231, -21.49019608])
    mode_x00  = np.array([0., -1.20337817, -2.88734466, -5.69412578, -5.77864306, -6.41033435, -4.87119596, -3.87443409, -4.55993979, -4.97101879, -6.33432908, -7.79768212, -9.04447682, -9.18571429, -11.50839771, -12.5, -12.12216085, -12.23255814, -13.76346154, -13.85483871, -17.10650922, -14.4075, -17.40120968, -16.8875, -22.14052174, -22.96875, -24.43778409, -24.89473684, -31.4, -32.05, -43.67428571, -53.3, -54.75535714, -49.22, -53.78809524, -58.12857143, -61.72554348, -68.5127451, -67.39454545, -67.38627451])
    mode_lo0  = np.array([0., -9.66312168e-01, -2.87479106e+00, -7.29536973e+00, -7.95834533e+00, -1.06151976e+01, -1.36872319e+01, -2.08717710e+01, -2.31630743e+01, -2.49369931e+01, -2.65471652e+01, -2.73668874e+01, -2.81333709e+01, -2.79327731e+01, -3.12690485e+01, -3.00142857e+01, -3.20612351e+01, -3.30651163e+01, -3.21614316e+01, -3.25419355e+01, -4.00362327e+01, -3.40200000e+01, -3.76151498e+01, -3.30041667e+01, -4.35780000e+01, -4.72875000e+01, -4.28068182e+01, -3.82052632e+01, -5.30021429e+01, -4.65833333e+01, -5.37314286e+01, 5.74000000e+01, -3.44464286e+01, -4.19400000e+01, 7.64392857e+01, -1.84028571e+02, -7.27240036e+01, 2.36645098e+02, -1.50630420e+02, 1.14704510e+03])
    mode_hi0  = np.array([0., 1.54954645, 3.32770232, 4.89327344, 3.69030019, 3.17477204, 2.02656062, 1.28055925, 1.14883498, 0.25796241, -0.24003661, -1.47913907, -2.15001566, -2.49747899, -2.74724953, -4.41714286, -3.36513478, -4.11860465, -5.34070513, -6.78387097, -5.66428571, -6.245, -6.60956221, -8.25416667, -9.15434783, -8.85625, -9.02215909, -7.74210526, -10.04071429, -12.725, -15.24285714, -16.8, -17.59107143, -16.64, -18.46666667, -19.41428571, -21.32961957, -24.65980392, -23.98048951, -24.63333333])
    mu_x00    = np.array([0., 0.00029715, 0.00087157, 0.00059894, 0.00026043, 0.00182371, 0.00057707, 0.00066578, -0.00011848, -0.00059347, -0.00040163, -0.00033113, -0.00185464, -0.00252101, 0.00337079, -0.00571429, 0.00640742, 0.00465116, 0.00683761, -0.00322581, 0.01002304, 0.005, 0.00213134, 0.02083333, 0.00869565, 0.00625, 0.03863636, 0., -0.00642857, -0.01666667, -0.03428571, -0.1, 0.00178571, -0.06, 0.05357143, 0.01428571, 0.03822464, 0.02156863, 0.01433566, 0.03333333])
    mu_lo0    = np.array([0., -0.21964342, -0.1893928, -0.45026492, -1.09734049, -1.70790274, -2.96855898, -4.61544607, -4.93513753, -5.60336301, -6.05451151, -6.59900662, -7.04603383, -7.1697479, -7.60229986, -8.23571429, -8.02544189, -8.40465116, -8.85769231, -9.5, -9.56664747, -8.9325, -9.56854839, -9.775, -10.83669565, -11.1125, -11.01761364, -9.78421053, -11.82452381, -14.54166667, -15.96571429, -17.05, -17.30535714, -17.08, -17.60119048, -18.2, -18.60235507, -20.3872549, -20.21643357, -20.50784314])
    mu_hi0    = np.array([0., -0.57546137, -0.50229283, -0.71856715, -1.56643187, -2.33799392, -3.7230101, -5.72343542, -6.05454926, -7.02611276, -7.5911653, -8.4513245, -9.04367481, -9.36554622, -9.92096208, -10.86571429, -10.67653557, -11.26976744, -11.81698718, -12.77096774, -13.00276498, -12.195, -13.20702765, -13.625, -15.2053913, -15.65625, -15.84602273, -14.62105263, -17.25809524, -20.38333333, -22.85285714, -24.8, -25.44821429, -24.9, -26.35238095, -27.58571429, -30.02871377, -34.83627451, -33.78818182, -34.24313725])
    med_x01   = np.array([0., -0.10423835, -0.20060066, -0.35712969, -0.36194218, -0.36504559, -0.29465695, -0.21211718, -0.25041482, -0.2578635, -0.30542526, -0.34072848, -0.38050125, -0.40252101, -0.43283591, -0.43857143, -0.44856385, -0.43023256, -0.45010684, -0.37419355, -0.43778802, -0.49, -0.43421659, -0.425, -0.32043478, -0.3625, -0.34261364, -0.78421053, -0.37166667, 0.31666667, 0.76428571, 1.25, 1.35714286, 1.28, 1.65357143, 1.98571429, 2.46757246, 4.26666667, 3.93629371, 4.18039216])
    med_lo1   = np.array([0., -0.15846106, -0.13888909, -0.43676572, -0.91382164, -1.34772036, -2.2479946, -3.37057257, -3.66958014, -4.02957468, -4.32505283, -4.58377483, -4.84233709, -4.95798319, -5.18655782, -5.44428571, -5.41411843, -5.59534884, -5.84850427, -5.96129032, -6.31434332, -6.1125, -6.28341014, -6.29583333, -6.69747826, -6.8875, -6.76789773, -6.52105263, -7.28095238, -7.775, -8.29142857, -8.6, -8.76428571, -8.42, -8.60595238, -8.68571429, -9.22952899, -8.38333333, -8.38272727, -8.17647059])
    med_hi1   = np.array([0., -0.59488583, -0.44634309, -0.57876065, -1.5014762, -2.33890578, -3.78509328, -5.82569907, -6.0593767, -7.07299703, -7.59196413, -8.47516556, -9.03924499, -9.28991597, -9.80181414, -10.79285714, -10.50132567, -11.08837209, -11.58344017, -12.61290323, -12.51900922, -11.6575, -12.64182028, -13.10833333, -14.55678261, -14.9, -14.97897727, -13.46842105, -15.97452381, -19.44166667, -21.45714286, -22.95, -23.30535714, -23.06, -23.97261905, -24.92857143, -25.78152174, -29.23823529, -28.80503497, -29.31960784])
    mode_x01  = np.array([0., -2.19533938e-01, -7.39874050e-01, -1.46268141e+00, -1.34183318e+00, -1.33708207e+00, -4.78398731e-01, -1.93075899e-02, 4.06317530e-03, 1.86547972e-01, 3.45820362e-01, 8.77483444e-01, 1.19317669e+00, 1.41848739e+00, 1.83096910e+00, 3.16571429e+00, 2.34954706e+00, 2.62093023e+00, 3.54871795e+00, 4.62258065e+00, 5.42476959e+00, 4.37750000e+00, 5.87039171e+00, 7.89583333e+00, 1.11186957e+01, 1.04187500e+01, 1.05400568e+01, 7.91578947e+00, 1.61621429e+01, 1.98166667e+01, 3.23457143e+01, 4.84000000e+01, 5.16178571e+01, 4.41400000e+01, 5.75119048e+01, 6.38285714e+01, 8.44162138e+01, 1.16050980e+02, 1.05390350e+02, 1.11249020e+02])
    mode_lo1  = np.array([0., 1.42217704e+00, 1.61731719e+00, 1.56915457e+00, 1.98197409e+00, 2.52887538e+00, 6.74525258e+00, 1.22810919e+01, 1.42843230e+01, 1.80713155e+01, 2.12498823e+01, 2.66231788e+01, 3.13593515e+01, 3.26504202e+01, 3.50149169e+01, 4.99928571e+01, 4.09904883e+01, 4.36627907e+01, 5.59903846e+01, 6.43870968e+01, 6.96858871e+01, 6.01350000e+01, 7.31752880e+01, 9.58958333e+01, 1.34087826e+02, 1.23731250e+02, 1.04453409e+02, 7.87315789e+01, -2.79214286e+01, 2.25525000e+02, 4.11300000e+02, 2.81985000e+03, 9.82232143e+02, 6.44920000e+02, 2.23775119e+03, -1.05378571e+03, 2.54829982e+02, 2.97942745e+03, -5.93335455e+02, 1.13853431e+04])
    mode_hi1  = np.array([0., -0.19895214, 0.56932248, 0.98378254, -0.44724886, -1.49787234, -4.29821053, -7.04926764, -7.23255599, -8.66122651, -9.29340089, -10.71324503, -11.53850564, -11.61848739, -12.04630735, -13.95714286, -12.51756518, -13.32790698, -14.6616453, -16.52903226, -15.0452765, -14.7725, -15.58306452, -17.10416667, -18.6246087, -18.30625, -18.36732955, -15.64210526, -19.21738095, -24.11666667, -26.89285714, -28.7, -29.34107143, -28.4, -30.06309524, -31.04285714, -32.23532609, -35.3745098, -34.87517483, -35.54509804])
    mu_x01    = np.array([0., -0.01693775, -0.01509655, -0.00764801, -0.03665688, -0.05319149, -0.0767023, -0.09201065, -0.10903816, -0.12720079, -0.17893366, -0.24933775, -0.30700501, -0.3512605, -0.43652271, -0.54571429, -0.54751436, -0.61162791, -0.69722222, -0.78387097, -0.88104839, -0.865, -1.04112903, -1.125, -1.37417391, -1.45, -1.61363636, -1.75263158, -2.05, -2.20833333, -2.93428571, -3.55, -3.69285714, -3.56, -3.93452381, -4.37142857, -5.14764493, -6.7127451, -6.39636364, -6.54901961])
    mu_lo1    = np.array([0., -0.2878949, -0.24326816, -0.48587883, -1.18890378, -1.84164134, -3.19854535, -4.94340879, -5.30135622, -6.0487636, -6.57040179, -7.19966887, -7.7127193, -7.88571429, -8.37331461, -9.11, -8.9172669, -9.38139535, -9.87382479, -10.67096774, -10.70985023, -10.0925, -10.84366359, -11.11666667, -12.30956522, -12.60625, -12.63295455, -11.3, -13.5, -16.60833333, -18.26714286, -19.55, -19.90892857, -19.72, -20.40357143, -21.18571429, -21.94003623, -24.44705882, -24.14055944, -24.52745098])
    mu_hi1    = np.array([0., -0.50639662, -0.44836733, -0.6812716, -1.47441653, -2.2, -3.48689051, -5.38695073, -5.6721632, -6.56379822, -7.05257311, -7.81788079, -8.33306704, -8.60504202, -9.08800328, -9.91285714, -9.70297172, -10.20232558, -10.69519231, -11.47096774, -11.72200461, -10.88, -11.73582949, -12.09583333, -13.49765217, -13.91875, -13.91022727, -12.73157895, -15.20380952, -17.88333333, -20.06142857, -21.75, -22.16607143, -21.56, -22.75119048, -23.72857143, -25.28387681, -28.51666667, -27.87538462, -28.2       ])
    med_x02   = np.array([0., -0.02655615, -0.04527656, -0.05899562, -0.03112736, 0.02340426, 0.07310926, 0.14327563, 0.17364788, 0.22571711, 0.30649548, 0.44370861, 0.53727444, 0.56386555, 0.74719686, 0.92714286, 0.84415599, 0.90465116, 1.08942308, 1.2516129, 1.47776498, 1.1725, 1.53479263, 1.62083333, 2.32478261, 2.3625, 2.51505682, 1.98947368, 3.25761905, 4.3, 6.10571429, 7.8, 8.14107143, 7.42, 8.56309524, 9.47142857, 10.6365942, 13.99313725, 13.35524476, 13.77647059])
    med_x02   = np.array([0., -0.02655615, -0.04527656, -0.05899562, -0.03112736, 0.02340426, 0.07310926, 0.14327563, 0.17364788, 0.22571711, 0.30649548, 0.44370861, 0.53727444, 0.56386555, 0.74719686, 0.92714286, 0.84415599, 0.90465116, 1.08942308, 1.2516129, 1.47776498, 1.1725, 1.53479263, 1.62083333, 2.32478261, 2.3625, 2.51505682, 1.98947368, 3.25761905, 4.3, 6.10571429, 7.8, 8.14107143, 7.42, 8.56309524, 9.47142857, 10.6365942, 13.99313725, 13.35524476, 13.77647059])
    med_lo2   = np.array([0., -0.10157961, 0.07652753, -0.06376411, -0.40039761, -0.65440729, -0.95177444, -1.22303595, -1.21035145, -1.1950544, -1.10033348, -1.00231788, -0.89578634, -0.88319328, -0.68339185, -0.50428571, -0.57159744, -0.51627907, -0.26944444, 0.10322581, 0.14406682, -0.31, 0.16883641, 0.2625, 1.14504348, 1.26875, 1.42926136, 0.41578947, 2.15785714, 4.35, 6.15428571, 7.8, 8.30357143, 7.94, 9.09404762, 10.11428571, 11.81304348, 16.27352941, 15.44111888, 15.96862745])
    med_hi2   = np.array([0., -0.54321239, -0.25407081, -0.28604008, -1.10184426, -1.85075988, -2.90203477, -4.52689747, -4.61149633, -5.45905045, -5.79821604, -6.56324503, -6.96723371, -7.22268908, -7.56215473, -8.39857143, -8.18289881, -8.71395349, -9.02094017, -9.83225806, -9.70368664, -9.13, -9.86947005, -10.30416667, -11.38043478, -11.6125, -11.72869318, -10.77894737, -12.45238095, -14.79166667, -16.26, -17.3, -17.57321429, -17.46, -18.15833333, -18.92857143, -19.41431159, -22.45, -22.10944056, -22.58235294])
    mode_x02  = np.array([0., 3.04191429e-02, -2.80414379e-01, -3.50725639e-01, -2.16963842e-01, -7.90273556e-03, 7.34064145e-01, 1.12383489e+00, 1.35101906e+00, 1.73531157e+00, 2.32847396e+00, 3.48675497e+00, 4.24477130e+00, 4.64537815e+00, 5.81217814e+00, 7.82142857e+00, 6.69107380e+00, 7.11162791e+00, 8.74722222e+00, 1.00516129e+01, 1.20863479e+01, 9.98750000e+00, 1.26587558e+01, 1.52333333e+01, 2.07633043e+01, 2.02312500e+01, 2.07838068e+01, 1.79947368e+01, 2.99976190e+01, 3.42083333e+01, 5.31857143e+01, 7.61000000e+01, 8.04821429e+01, 6.94800000e+01, 8.76166667e+01, 9.67714286e+01, 1.23695562e+02, 1.65532353e+02, 1.51718881e+02, 1.59103922e+02])
    mode_lo2  = np.array([0., 2.00231467e+00, 4.04266441e+00, 7.02842663e+00, 8.81712551e+00, 1.21337386e+01, 2.25290235e+01, 3.80203728e+01, 4.45807331e+01, 5.32475767e+01, 6.15795805e+01, 7.30798013e+01, 8.32544925e+01, 8.73168067e+01, 9.47015215e+01, 1.21822857e+02, 1.08409722e+02, 1.15025581e+02, 1.38047329e+02, 1.52425806e+02, 1.71640380e+02, 1.48790000e+02, 1.74753111e+02, 2.14466667e+02, 2.93697304e+02, 2.80031250e+02, 2.41555966e+02, 1.93547368e+02, 6.21761905e+00, 4.79025000e+02, 8.43945714e+02, 5.40105000e+03, 1.94441607e+03, 1.30242000e+03, 4.33716548e+03, -1.92120000e+03, 5.69825362e+02, 5.86411863e+03, -1.04793972e+03, 2.20181373e+04])
    mode_hi2  = np.array([0., 0.23736315, 2.32305533, 3.64247869, 2.7914139, 2.0893617, 1.09124436, 0.19653795, 0.672114, -0.0785361, -0.05363284, -1.11688742, -1.46721805, -1.65882353, -1.54037921, -3.11, -1.96681396, -2.62093023, -3.5275641, -4.8516129, -3.26912442, -4.0675, -4.25685484, -6.01666667, -6.49991304, -5.975, -6.18125, -5.20526316, -6.72047619, -8.775, -10.73285714, -11.8, -12.575, -11.94, -13.55952381, -14.44285714, -16.23623188, -19.7627451, -19.05713287, -19.74705882])
    mu_x02    = np.array([0., -0.02291211, -0.0206468, -0.01204792, -0.04674513, -0.07112462, -0.09335691, -0.11171771, -0.12555048, -0.15994065, -0.22501694, -0.31192053, -0.3901817, -0.43697479, -0.54626639, -0.67857143, -0.68729563, -0.76511628, -0.87190171, -0.99032258, -1.10927419, -1.0825, -1.3046659, -1.40416667, -1.73008696, -1.8125, -2.03380682, -2.21578947, -2.56952381, -2.78333333, -3.66428571, -4.45, -4.62678571, -4.48, -4.95119048, -5.48571429, -6.46240942, -8.49215686, -8.08755245, -8.29215686])
    mu_lo2    = np.array([0., -0.28747263, -0.24213657, -0.48472702, -1.18588244, -1.83556231, -3.19152561, -4.9322237, -5.28982317, -6.03461919, -6.55071858, -7.17417219, -7.6793703, -7.84285714, -8.33065309, -9.06, -8.85841803, -9.31395349, -9.79273504, -10.57741935, -10.59913594, -9.985, -10.71342166, -10.97083333, -12.1386087, -12.39375, -12.39573864, -11.03157895, -13.25238095, -16.33333333, -17.91142857, -19.15, -19.45714286, -19.26, -19.84047619, -20.58571429, -20.83043478, -22.94509804, -22.81944056, -23.23921569])
    mu_hi2    = np.array([0., -0.50584923, -0.44741869, -0.67995853, -1.47146783, -2.19422492, -3.48025376, -5.37736352, -5.66033045, -6.54777448, -7.0319951, -7.79172185, -8.2999812, -8.56890756, -9.03768141, -9.85714286, -9.63683164, -10.11860465, -10.60598291, -11.36774194, -11.60449309, -10.75, -11.58248848, -11.925, -13.29330435, -13.7125, -13.63380682, -12.41578947, -14.88714286, -17.53333333, -19.63571429, -21.25, -21.5875, -20.96, -22.04166667, -22.92857143, -23.83025362, -26.27745098, -25.94286713, -26.27647059])
  elif data == 'again':
    skew,med_x00   = plotTests(0,data='med',x0='med', output=True,pdf='med_x00.pdf')
  # print('skew:    ', skew)
  # print('med_x00 :', med_x00 )
    skew,med_lo0   = plotTests(0,data='lo' ,x0='med', output=True,pdf='med_lo0.pdf')
  # print('med_lo0: ', med_lo0)
    skew,med_hi0   = plotTests(0,data='hi' ,x0='med', output=True,pdf='med_hi0.pdf')
  # print('med_hi0: ', med_hi0)
    skew,mode_x00  = plotTests(0,data='med',x0='mode',output=True,pdf='mode_x00.pdf')
  # print('mode_x00 :   ', mode_x00 )
    skew,mode_lo0  = plotTests(0,data='lo' ,x0='mode',output=True,pdf='mode_lo0.pdf')
  # print('mode_lo0:    ', mode_lo0)
    skew,mode_hi0  = plotTests(0,data='hi' ,x0='mode',output=True,pdf='mode_hi0.pdf')
  # print('mode_hi0:    ', mode_hi0)
    skew,mu_x00    = plotTests(0,data='med',x0='mu',  output=True,pdf='mu_x00.pdf')
  # print('mu_x00 : ', mu_x00 )
    skew,mu_lo0    = plotTests(0,data='lo' ,x0='mu',  output=True,pdf='mu_lo0.pdf')
  # print('mu_lo0:  ', mu_lo0)
    skew,mu_hi0    = plotTests(0,data='hi' ,x0='mu',  output=True,pdf='mu_hi0.pdf')
  # print('mu_hi0:  ', mu_hi0)
    skew,med_x01   = plotTests(1,data='med',x0='med', output=True,pdf='med_x01.pdf')
  # print('med_x01 :    ', med_x01 )
    skew,med_lo1   = plotTests(1,data='lo' ,x0='med', output=True,pdf='med_lo1.pdf')
  # print('med_lo1: ', med_lo1)
    skew,med_hi1   = plotTests(1,data='hi' ,x0='med', output=True,pdf='med_hi1.pdf')
  # print('med_hi1: ', med_hi1)
    skew,mode_x01  = plotTests(1,data='med',x0='mode',output=True,pdf='mode_x01.pdf')
  # print('mode_x01 :   ', mode_x01 )
    skew,mode_lo1  = plotTests(1,data='lo' ,x0='mode',output=True,pdf='mode_lo1.pdf')
  # print('mode_lo1:    ', mode_lo1)
    skew,mode_hi1  = plotTests(1,data='hi' ,x0='mode',output=True,pdf='mode_hi1.pdf')
  # print('mode_hi1:    ', mode_hi1)
    skew,mu_x01    = plotTests(1,data='med',x0='mu',  output=True,pdf='mu_x01.pdf')
  # print('mu_x01 : ', mu_x01 )
    skew,mu_lo1    = plotTests(1,data='lo' ,x0='mu',  output=True,pdf='mu_lo1.pdf')
  # print('mu_lo1:  ', mu_lo1)
    skew,mu_hi1    = plotTests(1,data='hi' ,x0='mu',  output=True,pdf='mu_hi1.pdf')
  # print('mu_hi1:  ', mu_hi1)
    skew,med_x02   = plotTests(2,data='med',x0='med', output=True,pdf='med_x02.pdf')
  # print('med_x02 :    ', med_x02 )
    skew,med_x02   = plotTests(2,data='med',x0='med', output=True,pdf='med_x02.pdf')
  # print('med_x02 :    ', med_x02 )
    skew,med_lo2   = plotTests(2,data='lo' ,x0='med', output=True,pdf='med_lo2.pdf')
  # print('med_lo2: ', med_lo2)
    skew,med_hi2   = plotTests(2,data='hi' ,x0='med', output=True,pdf='med_hi2.pdf')
  # print('med_hi2: ', med_hi2)
    skew,mode_x02  = plotTests(2,data='med',x0='mode',output=True,pdf='mode_x02.pdf')
  # print('mode_x02 :   ', mode_x02 )
    skew,mode_lo2  = plotTests(2,data='lo' ,x0='mode',output=True,pdf='mode_lo2.pdf')
  # print('mode_lo2:    ', mode_lo2)
    skew,mode_hi2  = plotTests(2,data='hi' ,x0='mode',output=True,pdf='mode_hi2.pdf')
  # print('mode_hi2:    ', mode_hi2)
    skew,mu_x02    = plotTests(2,data='med',x0='mu',  output=True,pdf='mu_x02.pdf')
  # print('mu_x02 : ', mu_x02 )
    skew,mu_lo2    = plotTests(2,data='lo' ,x0='mu',  output=True,pdf='mu_lo2.pdf')
  # print('mu_lo2:  ', mu_lo2)
    skew,mu_hi2    = plotTests(2,data='hi' ,x0='mu',  output=True,pdf='mu_hi2.pdf')
  # print('mu_hi2:  ', mu_hi2)

  plt.clf()
  plt.xlim([0,10])
  plt.ylim([-30,10])
  plt.xlabel('Skewness')
  plt.ylabel('Offset in %')

  lwx0 = 6
  lws1 = .3
  lws2 = 3
# plt.plot(skew,med_x00  ,'b',  lw=lwx0,label=r'$0: x_0\equiv\,\mathrm{median;} \,x_0$')
# plt.plot(skew,med_lo0  ,'b',  lw=lws1,label=r'$0: x_0\equiv\,\mathrm{median;} \,\sigma_{-}$')
  plt.plot(skew,med_hi0  ,'b',  lw=lws2,label=r'$0: x_0\equiv\,\mathrm{median;} \,\sigma_{+}$')
# plt.plot(skew,mu_x00   ,'r',  lw=lwx0,label=r'$0: x_0\equiv\mu;               \,x_0$')
# plt.plot(skew,mu_lo0   ,'r',  lw=lws1,label=r'$0: x_0\equiv\mu;               \,\sigma_{-}$')
  plt.plot(skew,mu_hi0   ,'r',  lw=lws2,label=r'$0: x_0\equiv\mu;               \,\sigma_{+}$')
# plt.plot(skew,mode_x00 ,'g',  lw=lwx0,label=r'$0: x_0\equiv\,\mathrm{mode;}   \,x_0$')
# plt.plot(skew,mode_lo0 ,'g',  lw=lws1,label=r'$0: x_0\equiv\,\mathrm{mode;}   \,\sigma_{-}$')
  plt.plot(skew,mode_hi0 ,'g',  lw=lws2,label=r'$0: x_0\equiv\,\mathrm{mode;}   \,\sigma_{+}$')

# plt.plot(skew,med_x01  ,'b--',lw=lwx0,label=r'$1: x_0\equiv\,\mathrm{median;} \,x_0$')
# plt.plot(skew,med_lo1  ,'b--',lw=lws1,label=r'$1: x_0\equiv\,\mathrm{median;} \,\sigma_{-}$')
  plt.plot(skew,med_hi1  ,'b--',lw=lws2,label=r'$1: x_0\equiv\,\mathrm{median;} \,\sigma_{+}$')
# plt.plot(skew,mu_x01   ,'r--',lw=lwx0,label=r'$1: x_0\equiv\mu;               \,x_0$')
# plt.plot(skew,mu_lo1   ,'r--',lw=lws1,label=r'$1: x_0\equiv\mu;               \,\sigma_{-}$')
  plt.plot(skew,mu_hi1   ,'r--',lw=lws2,label=r'$1: x_0\equiv\mu;               \,\sigma_{+}$')
# plt.plot(skew,mode_x01 ,'g--',lw=lwx0,label=r'$1: x_0\equiv\,\mathrm{mode;}   \,x_0$')
# plt.plot(skew,mode_lo1 ,'g--',lw=lws1,label=r'$1: x_0\equiv\,\mathrm{mode;}   \,\sigma_{-}$')
  plt.plot(skew,mode_hi1 ,'g--',lw=lws2,label=r'$1: x_0\equiv\,\mathrm{mode;}   \,\sigma_{+}$')

# plt.plot(skew,med_x02  ,'b:', lw=lwx0,label=r'$2: x_0\equiv\,\mathrm{median;} \,x_0$')
# plt.plot(skew,med_lo2  ,'b:', lw=lws1,label=r'$2: x_0\equiv\,\mathrm{median;} \,\sigma_{-}$')
  plt.plot(skew,med_hi2  ,'b:', lw=lws2,label=r'$2: x_0\equiv\,\mathrm{median;} \,\sigma_{+}$')
# plt.plot(skew,mu_x02   ,'r:', lw=lwx0,label=r'$2: x_0\equiv\mu;               \,x_0$')
# plt.plot(skew,mu_lo2   ,'r:', lw=lws1,label=r'$2: x_0\equiv\mu;               \,\sigma_{-}$')
  plt.plot(skew,mu_hi2   ,'r:', lw=lws2,label=r'$2: x_0\equiv\mu;               \,\sigma_{+}$')
# plt.plot(skew,mode_x02 ,'g:', lw=lwx0,label=r'$2: x_0\equiv\,\mathrm{mode;}   \,x_0$')
# plt.plot(skew,mode_lo2 ,'g:', lw=lws1,label=r'$2: x_0\equiv\,\mathrm{mode;}   \,\sigma_{-}$')
  plt.plot(skew,mode_hi2 ,'g:', lw=lws2,label=r'$2: x_0\equiv\,\mathrm{mode;}   \,\sigma_{+}$')

  plt.legend(loc=3,fontsize=9,frameon=False,handlelength=2.5,ncol=3)
#------------------------------------------------------------------------------

def show_skewness_example(n=1,skew_range=[0,10],png=False):
  """
  Draw 'n' figures with skewness in 'skew_range' range.
  The plot window may be resized manually first.

  Example:
  --------
  To create 5 png files with skewness in [0,1]:
  >>> show_skewness_example(n=5,skew_range=[0,1],png=True)
  
  To merge them in a column (use +append for row):
  > convert -append *.png skew.png
  """
  import formats as f
  x = np.linspace(-20,200,220000)

  i = 0
  while i<n:
    P = getRandomDistr(x,allowInverted=False)[0]
    S = skewness(P,x)
    if not (skew_range[0] <= S <= skew_range[1]): continue
    i += 1
    print(i)
    s = f.fflt(S)
    x0,s1,s2 = f.fflt(plotCI(x,P,plotit=False,output=True,verb=False)[0:3], d=1)
    ind = np.where(P>max(P)/100.)                 #Plot only interesting part
    plt.clf()
    plt.rcParams.update({'font.size': 18})
    ax = plt.subplot(1,1,1)
    ax.set_title('Skewness = '+s)
    ax.plot(x[ind],P[ind],lw=3,label=r'$'+x0+'_{-'+s1+'}^{+'+s2+'}$')
    ax.text(.6,.8,r'$'+x0+'_{-'+s1+'}^{+'+s2+'}$',transform=ax.transAxes,
      fontsize=30)
    if png: plt.savefig(s+'.png', bbox_inches='tight')
#------------------------------------------------------------------------------

def x0s1s2(x0,lower,upper):
  """
  For a central, lower, and upper values x0, lo, and hi, respectively,
  return x0 and the differences between lo,hi and x0.

  Example
  -------
  >>> x0,siglo,sighi = x0s1s2(x0,lo,hi)
  """
  return x0-lo, hi-x0
#------------------------------------------------------------------------------

def testCLT():

    """
    Test that addition of multiple asymmetric PDFs follow the Central Limit
    Theorem.
    """

    x0 = 0.
    s1 = 2.
    s2 = 3.
  # x0,s1,s2 = -0.000135540, 1.999178879, +2.990626866 # skewGauss(x,-2.744594,4.069,5.21)
  # x0,s1,s2 = 0.000000006, 1.990212159, +3.003211291  # Frechet(x,20,41,-41.7582383)

    nmax    = 25
    nPDFs   = range(1,nmax+1)#np.array(range(n)) + 1
    n       = len(nPDFs)
    x0tot0  = np.zeros_like(nPDFs, dtype='f8')
    x0tot1  = np.zeros_like(nPDFs, dtype='f8')
    x0tot2  = np.zeros_like(nPDFs, dtype='f8')
    s1tot0  = np.zeros_like(nPDFs, dtype='f8')
    s1tot1  = np.zeros_like(nPDFs, dtype='f8')
    s1tot2  = np.zeros_like(nPDFs, dtype='f8')
    s2tot0  = np.zeros_like(nPDFs, dtype='f8')
    s2tot1  = np.zeros_like(nPDFs, dtype='f8')
    s2tot2  = np.zeros_like(nPDFs, dtype='f8')
    s1true1 = np.zeros_like(nPDFs, dtype='f8')
    s2true1 = np.zeros_like(nPDFs, dtype='f8')
    s1true2 = np.zeros_like(nPDFs, dtype='f8')
    s2true2 = np.zeros_like(nPDFs, dtype='f8')
    s1true3 = np.zeros_like(nPDFs, dtype='f8')
    s2true3 = np.zeros_like(nPDFs, dtype='f8')
    s1true4 = np.zeros_like(nPDFs, dtype='f8')
    s2true4 = np.zeros_like(nPDFs, dtype='f8')
    s1true5 = np.zeros_like(nPDFs, dtype='f8')
    s2true5 = np.zeros_like(nPDFs, dtype='f8')
    medtrue1= np.zeros_like(nPDFs, dtype='f8')
    medtrue2= np.zeros_like(nPDFs, dtype='f8')
    medtrue3= np.zeros_like(nPDFs, dtype='f8')
    medtrue4= np.zeros_like(nPDFs, dtype='f8')
    medtrue5= np.zeros_like(nPDFs, dtype='f8')

    mkNew = False
    if mkNew:
        x  = np.linspace(-50,70,10000)
        P1 = skewGauss(x,-2.744594,4.069,5.21)
        P2 = Frechet(x,20,41,-41.7582383)
        P3 = lognorm(x+6.0014,1.792,.40555269)
        P4 = loglogistic(x+5.9801,5.982,4.096)
        P5 = Weibull(x+2.36,4.255,1.526)
        nsamp = 10000#000
        R1 = np.random.choice(x,(nsamp,n),p=P1/sum(P1))
        R2 = np.random.choice(x,(nsamp,n),p=P2/sum(P2))
        R3 = np.random.choice(x,(nsamp,n),p=P3/sum(P3))
        R4 = np.random.choice(x,(nsamp,n),p=P4/sum(P4))
        R5 = np.random.choice(x,(nsamp,n),p=P5/sum(P5))

        for i,nn in enumerate(nPDFs):
            x0tot0[i],s1tot0[i],s2tot0[i] = add_asym([x0]*nn,[s1]*nn,[s2]*nn,order=0,ohwell=True)
            x0tot1[i],s1tot1[i],s2tot1[i] = add_asym([x0]*nn,[s1]*nn,[s2]*nn,order=1)
            x0tot2[i],s1tot2[i],s2tot2[i] = add_asym([x0]*nn,[s1]*nn,[s2]*nn,order=2)
            MCsum1,dum           = np.histogram(np.sum(R1[:,:i+1],axis=1),bins=len(x),density=True)
            MCsum2,dum           = np.histogram(np.sum(R2[:,:i+1],axis=1),bins=len(x),density=True)
            MCsum3,dum           = np.histogram(np.sum(R3[:,:i+1],axis=1),bins=len(x),density=True)
            MCsum4,dum           = np.histogram(np.sum(R4[:,:i+1],axis=1),bins=len(x),density=True)
            MCsum5,dum           = np.histogram(np.sum(R5[:,:i+1],axis=1),bins=len(x),density=True)
            med1,lo1,hi1           = wp(x,MCsum1,[0.5,0.1587,0.8413])
            med2,lo2,hi2           = wp(x,MCsum2,[0.5,0.1587,0.8413])
            med3,lo3,hi3           = wp(x,MCsum3,[0.5,0.1587,0.8413])
            med4,lo4,hi4           = wp(x,MCsum4,[0.5,0.1587,0.8413])
            med5,lo5,hi5           = wp(x,MCsum5,[0.5,0.1587,0.8413])
            s1true1[i],s2true1[i] = med1-lo1, hi1-med1
            s1true2[i],s2true2[i] = med2-lo2, hi2-med2
            s1true3[i],s2true3[i] = med3-lo3, hi3-med3
            s1true4[i],s2true4[i] = med4-lo4, hi4-med4
            s1true5[i],s2true5[i] = med5-lo5, hi5-med5
            medtrue1[i] = med1
            medtrue2[i] = med2
            medtrue3[i] = med3
            medtrue4[i] = med4
            medtrue5[i] = med5

        rat1 = s2true1/s1true1
        rat2 = s2true2/s1true2
        rat3 = s2true3/s1true3
        rat4 = s2true4/s1true4
        rat5 = s2true5/s1true5
        np.savetxt('testCLT_skewGauss.dat',  rat1)
        np.savetxt('testCLT_Frechet.dat',    rat2)
        np.savetxt('testCLT_lognorm.dat',    rat3)
        np.savetxt('testCLT_loglogistic.dat',rat4)
        np.savetxt('testCLT_Weibull.dat',    rat5)
    else:
        for i,nn in enumerate(nPDFs):
            x0tot0[i],s1tot0[i],s2tot0[i] = add_asym([x0]*nn,[s1]*nn,[s2]*nn,order=0,ohwell=True)
            x0tot1[i],s1tot1[i],s2tot1[i] = add_asym([x0]*nn,[s1]*nn,[s2]*nn,order=1)
            x0tot2[i],s1tot2[i],s2tot2[i] = add_asym([x0]*nn,[s1]*nn,[s2]*nn,order=2)
        rat1 = np.loadtxt('testCLT_skewGauss.dat')
        rat2 = np.loadtxt('testCLT_Frechet.dat')
        rat3 = np.loadtxt('testCLT_lognorm.dat')
        rat4 = np.loadtxt('testCLT_loglogistic.dat')
        rat5 = np.loadtxt('testCLT_Weibull.dat')

    plt.clf()

  # ARGH MEDIAN DOESN'T WORK
  #
  # plt.plot(nPDFs,x0tot0,'r-o',label='order = 0')
  # plt.fill_between(nPDFs,x0tot0-s1tot0,x0tot0+s2tot0,alpha=.25,color='r')
  # plt.plot(nPDFs,x0tot1,'g-o',label='order = 1')
  # plt.fill_between(nPDFs,x0tot1-s1tot1,x0tot1+s2tot1,alpha=.25,color='g')
  # plt.plot(nPDFs,x0tot2,'y-o',label='order = 2')
  # plt.fill_between(nPDFs,x0tot2-s1tot2,x0tot2+s2tot2,alpha=.25,color='y')
  # plt.plot(nPDFs,medtrue1,'b-o',lw=.5)
  # plt.fill_between(nPDFs,medtrue1-s1true1,medtrue1+s2true1,alpha=.10,color='b')
  # plt.plot(nPDFs,medtrue2,'b-o',lw=.5)
  # plt.fill_between(nPDFs,medtrue2-s1true2,medtrue2+s2true2,alpha=.10,color='b')
  # plt.plot(nPDFs,medtrue3,'b-o',lw=.5)
  # plt.fill_between(nPDFs,medtrue3-s1true3,medtrue3+s2true3,alpha=.10,color='b')
  # plt.plot(nPDFs,medtrue4,'b-o',lw=.5)
  # plt.fill_between(nPDFs,medtrue4-s1true4,medtrue4+s2true4,alpha=.10,color='b')
  # plt.plot(nPDFs,medtrue5,'b-o',lw=.5)
  # plt.fill_between(nPDFs,medtrue5-s1true5,medtrue5+s2true5,alpha=.10,color='b')

    plt.xlim([0,nmax])
    plt.ylim([0.7,1.7])
    plt.gca().set_xticks(range(0,26,5))
    plt.xlabel('# of PDFs added')
    plt.ylabel(r'$\sigma_{+}\,/\,\sigma_{-}$')
    plt.plot(nPDFs,s2tot0/s1tot0,'r-o',lw=2,markersize=3,label='"Usual" approach')
    plt.plot(nPDFs,s2tot1/s1tot1,'g-o',lw=2,markersize=3,label='Piecewise linear',zorder=5)
    plt.plot(nPDFs,s2tot2/s1tot2,'y-o',lw=2,markersize=3,label='Quadratic',zorder=10)
    plt.plot(nPDFs,rat1,'b-o',lw=.5,markersize=1,label='True convolutions')
    plt.plot(nPDFs,rat2,'b-o',lw=.5,markersize=1)
    plt.plot(nPDFs,rat3,'b-o',lw=.5,markersize=1)
    plt.plot(nPDFs,rat4,'b-o',lw=.5,markersize=1)
    plt.plot(nPDFs,rat5,'b-o',lw=.5,markersize=1)
    plt.legend(frameon=False)

    plt.savefig('testCLT.pdf', bbox_inches='tight')
#------------------------------------------------------------------------------
