# Code for the estimation of translational diffusion coefficients from simulation data
# version 1.0 (06/02/2020)
# Jakob Tom√s Bullerjahn
# Soeren von Buelow (soeren.buelow@biophys.mpg.de)
# Gerhard Hummer

# Please read and cite the paper: 
# J. Bullerjahn, S. v. Buelow, G. Hummer: Optimal estimates of diffusion coefficients from molecular dynamics simulations, Journal of Chemical Physics XXX, YYYYY (2020).

# INPUT FOR CLASS Dfit.Dcov():
#
# REQUIRED:
# fz (str): Filename of input trajectory. Rows: Timeseries. Columns: Dimensions. Use no header. The number of dimensions in the trajectory is inferred from the number of columns. Length unit: nm, time unit: ps
# OPTIONAL:
# dt (float): Timestep, in units of ps (default: 1.0).
# m (int): Number of MSD values to consider (default: 20).
# tmin (int): Minimum timestep (default: 1).
# tmax (int): Maximum timestep (default: 100).
# d2max (float): Convergence criterion for GLS iteration (default: 1e-10).
# nitmax (int): Maximum number of iterations in GLS procedure (default: 100).
# nseg (int): Number of segments (default: N / (100*tmax)).
# fout (str): Name for output files w/o extension (default 'D_analysis').
# imgfmt (str): Output format for plot. Choose 'pdf' or 'png' (default: 'pdf').

# OUTPUT
#
# D_analysis.dat: Summary of output, including diffusion coefficient and Q-factor analysis.
# D_analysis.pdf: (or .png) Output plots. Use this to assess which diffusion coefficient to select. Refer to Figs. 2 and 4 in the paper.
# Aside from the output in (by default) D_analysis.dat and D_analysis.pdf, the diffusion coefficients and Q values are stored in the class:
# res.D: Optimal diffusion coefficient estimates per timestep
# res.Dstd: Estimated standard deviation of res.D per timestep
# res.Dempstd: Empirical standard deviation of res.D per timestep (should be close to res.Dstd if the diffusive model is appropriate)
# res.q_m: Mean quality factor per timestep
# res.q_std: Standard deviation of quality factor per timestep

# INPUT FOR analysis():
# tc (int): 'Timestep chosen' for the diffusion coefficient. tc is connected to the timestep in ps via dt*(tc+tmin-1)
# The analysis can be repeated with different values of tc. A red vertical line indicates the chosen timestep in the plot.

# INPUT FOR finite_size_correction():
# tc (int): Use tc determined in the previous analysis step
# eta (float): Viscosity in units of Pa*s
# L (float): Edge length of cubic (!) simulation box in nm
# T (float): Temperatur in Kelvin
# boxtype (str): Only 'cubic' currently allowed.

# EXAMPLE
#
# import Dfit
# res = Dfit.Dcov(m=20,fz='mytrajectory.dat',tmin=1,tmax=100,nseg=150)
# res.run_Dfit()
# res.analysis(tc=10)
# res.finite_size_correction(L=7.5, eta=0.9e-3, tc=10)
#

import numpy as np
import numba as nb
from scipy.special import gammainc
import sys
import matplotlib.pyplot as plt
import progressbar as bar

# =====================================================

### Covariance calculation ### 

@nb.jit(nopython=True)
def setupc(m,n):
    """
    Setup covariance matrix for MSD(k)=<(z(i+k)-z(i))^2> 
    Eq. 8b expression in [ brackets ] (excluding sig^4/3)
    """
    c = np.zeros((m,m))

    for i in range(1,m+1):
        for j in range(1,m+1):

            # Heaviside
            if (i + j - n - 2 >= 0):
                heav = 1.0
            else:
                heav = 0.0

            c[i-1,j-1] = (
                2.*min(i,j) * (1.+3.*i*j - min(i,j)**2) / (n - min(i,j) +1) 
                + (min(i,j)**2 - min(i,j)**4) / ((n-i+1.)*(n-j+1.))
                + heav * ((n+1.-i-j)**4 - (n+1.-i-j)**2) / ((n-i+1.)*(n-j+1.))
                )
    return c

@nb.jit(nopython=True)
def calc_cov(n,m,c,a2,s2):
    """ Eq. 8a, with factor s^4/3 from Eq. 8b """
    cov = np.zeros((m,m))
    for i in range(1,m):
        for j in range(i+1,m+1):
            cov[i-1,j-1] = c[i-1,j-1] * s2**2 / 3. + add_to_cov(a2,s2,n,i,j)

    cov += cov.T
    for i in range(1,m+1):
        cov[i-1,i-1] = c[i-1,i-1] * s2**2 / 3. + add_to_cov(a2,s2,n,i,i)
    return(cov)

@nb.jit(nopython=True)
def add_to_cov(a2,s2,n,i,j):
    """ Noise contribution Eq. 8a to covariance matrix element ij. """
    dirac = 1. if i==j else 0.
    add_cov = (
        ((1.+dirac)*a2**2 + 4. * a2 * s2 * min(i,j)) / (n-min(i,j)+1.) 
        + a2**2 * max(n-i-j+1.,0.) / ((n-i+1.) * (n-j+1.))
            )
    return add_cov

@nb.jit(nopython=True)
def inv_mat(A):
    cinv = np.linalg.inv(A)
    return cinv

# GLS calculation

def calc_gls(n,m,v,d2max,nitmax,step):
    a2est, s2est, s2varest = gls_closed(n,v)
    a2, s2, nit = gls_iter(n,m,v,a2est,s2est,d2max,nitmax)

    if nit >= nitmax:
        print('WARNING: Optimizer did not converge. Falling back to M=2.')
        a2, s2 = a2est, s2est # use M=2 result

    # a2 /= step
    # s2 /= step

    return a2, s2

# CHI2 and Q
@nb.jit(nopython=True)
def calc_chi2(m,a2_3D,s2_3D,msds_3D,cinv,ndim):
    chi2 = 0.0
    for i in range(m):
        for j in range(m):
            chi2 += (msds_3D[i] - a2_3D - s2_3D * (i+1.)) * (
                cinv[i,j] * (msds_3D[j] - a2_3D - s2_3D * (j+1.))
                )
    chi2 = chi2*ndim
    return chi2

### Closed-form GLS ###

def gls_closed(n,v):
    """
    Closed-form GLS estimation of offset a^2 and variance sigma^2.
    """

    # c = setupc(2,n)
    c = setupc(2,n) # setups of cov matrix

    s2 = v[1]-v[0]
    a2 = 2*v[0] - v[1]
    
    cov = calc_cov(n,2,c,a2,s2)
    cinv = inv_mat(cov)
 
    # a2var = abs(4. * cov[0,0] - 4. * cov[0,1] + cov[1,1])
    s2var = abs(cov[0,0] - 2. * cov[0,1] + cov[1,1])

    return a2, s2, s2var

### Iterative GLS solver ###

def gls_iter(n,m,v,a2,s2,d2max,nitmax):
    """
    Iteratively optimize offset a^2 and variance sigma^2
    of fit to MSD for Gaussian random walk with noise
    (model: MSD(i)=a^2+i*sigma^2).
    """

    nit = 0
    d2 = d2max + 1.e5

    c = setupc(m,n) # setups of cov matrix

    while (nit < nitmax) and (d2 > d2max):
        nit = nit+1
        cov = calc_cov(n,m,c,a2,s2)

        # calculate inverse covariance
        cinv = inv_mat(cov)

        # estimate new values for a^2 and sigma^2
        a2, s2, d2 = calc_a2s2(m, v, cinv, a2, s2)

    return a2, s2, nit

@nb.jit(nopython=True)
def calc_a2s2(m,v,cinv, a2, s2):
    A = 0.0
    B = 0.0
    C = 0.0
    D = 0.0
    E = 0.0

    for i in range(m):
        for j in range(m):
            A += cinv[i,j]
            B += (i+1) * cinv[i,j]
            C += (i+1) * (j+1) * cinv[i,j]
            D += v[i] * cinv[i,j]
            E += (i+1) * v[j] * cinv[i,j]

    denom = A * C - B**2

    a2n = (C*D-B*E) / denom
    s2n = (A*E-B*D) / denom

    d2=(s2n-s2)**2+(a2n-a2)**2
    return a2n, s2n, d2

@nb.jit(nopython=True)
def calc_var(n,m,a2,s2):

    c = setupc(m,n)
    cov = calc_cov(n,m,c,a2,s2)
    cinv = inv_mat(cov)

    A = 0.0
    B = 0.0
    C = 0.0

    for i in range(m):
        for j in range(m):
            A += cinv[i,j]
            B += (i+1) * cinv[i,j]
            C += (i+1) * (j+1) * cinv[i,j]
    denom = A * C - B**2
    # a2var = C / denom
    s2var = A / denom
    return s2var

def eval_vars(n,m,a2m,s2m,ndim,step):
    """ a2 and s2 are means across segments, 
    but still per dim """
    s2var = 0.0
    for d in range(ndim):
        s2var += calc_var(n,m,a2m[d],s2m[d]) # using mean across segments but per dim
    # s2var /= step**2
    return s2var

def calc_q(n,m,a2_3D,s2_3D,msds_3D,a2full_3D,s2full_3D,ndim):
    """ Q per segment (Eq. 22). Use best a2 and s2 estimates from 
    full trajectory for cov and cinv. """
    
    c = setupc(m,n) # setups of cov matrix
    cov = calc_cov(n,m,c,a2full_3D,s2full_3D)
    cinv = inv_mat(cov)

    chi2 = calc_chi2(m,a2_3D,s2_3D,msds_3D,cinv,ndim)

    if chi2 <= 0:
        q = 1.0
    else:
        q = 1-gammainc( (m-2)/2.,chi2/2.) # goodness-of-fit Q
    return q

def compute_MSD_1D_via_correlation(x):
    """
    One-dimensional MSD calculated via FFT-based auto-correlation.
    """
    corrx  = compute_correlation_via_fft(x)
    nt     = len(x)
    dsq    = x**2
    sumsq  = 2*np.sum(dsq)
    msd    = np.zeros((nt))
    msd[0] = 0.
    for m in range(1,nt):
        sumsq  = sumsq - dsq[m-1]-dsq[nt-m]
        msd[m] = sumsq/(nt-m) - 2*corrx[m]
    return msd

def compute_correlation_via_fft(x, y=None):
    """
    Correlation of two arrays calculated via FFT.
    """
    x   = np.array(x)
    l   = len(x)
    xft = np.fft.fft(x, 2*l)

    if y is None:
        yft = xft
    else:
        y   = np.array(y)
        yft = np.fft.fft(y, 2*l)
    corr    = np.real(np.fft.ifft(np.conjugate(xft)*yft))
    norm    = l - np.arange(l)
    corr    = corr[:l]/norm
    return corr

# ================================

class Dcov():

    def __init__(self, fz=None, m=20,tmin=1,tmax=100,dt=1.0,d2max=1e-10,nitmax=100,
    nseg=None,imgfmt = 'pdf',fout = 'D_analysis'):

        self.fz = fz
        if type(self.fz) in (list, tuple):
            self.multi = True
            print('Analyzing trajectories of multiple molecules from the same simulation.')
        else:
            print('Analyzing single trajectory.')
        self.dt = dt # Trajecotory timestep in ps
        self.m = m
        self.tmin = tmin
        self.tmax = tmax
        self.d2max = d2max
        self.nitmax = nitmax   

        if imgfmt not in ['pdf','png']:
            raise TypeError("Error! Choose 'pdf' or 'png' as output format.")
        self.imgfmt = imgfmt
        self.fout = fout

        if self.multi:
            self.zs = [np.loadtxt(f).T for f in self.fz]
            self.z = self.zs[0]  # This is only to determine ndim and n
        else:
            self.z = np.loadtxt(self.fz).T # read in timeseries (rows) for each dimensions (columns)

        if len(self.z.shape) > 1: # 2D data or more
            self.ndim = self.z.shape[0] # number of dimensions
            self.n = self.z.shape[1] - 1 # length of timeseries N+1
        else:
            self.ndim = 1
            self.n = self.z.shape[0] - 1 # length of timeseries N+1
        print('N = {}'.format(self.n))
        print('ndim = {}'.format(self.ndim))
        if self.multi:
            self.nseg = len(self.fz) # number of individual molecules
            self.nperseg = self.n # all molecules from trajectory with same length
        else:
            self.nseg = int((self.n+1) / (100. * self.tmax)) # number of segments
            if nseg != None: # given nseg
                if nseg > self.nseg: # compare if given nseg is over max nseg
                    print("Warning, too many segments chosen, falling back to nseg = {}".format(self.nseg))
                else:
                    self.nseg = nseg
            self.nperseg = int((self.n+1) / self.nseg) - 1 # length of segment timeseries Nperseg+1
            if self.nseg == 0:
                raise ValueError('Timeseries too short! Reduce tmax')
            self.a2full = np.zeros((self.tmax-self.tmin+1,self.ndim)) # full trajectory, per dim
            self.s2full = self.a2full.copy() # full trajectory, per dim

        if self.m > self.nperseg:
            self.m = self.nperseg # force self.m to not be larger than Nperseg

        self.a2 = np.zeros((self.tmax-self.tmin+1,self.nseg,self.ndim))  # per segment and dims
        self.s2 = self.a2.copy() # per segment and dims

        self.s2var = np.zeros((self.tmax-self.tmin+1)) # mean across all segments and dims
        self.q = np.zeros((self.tmax-self.tmin+1,self.nseg)) # per segment, mean across dims

    def run_Dfit(self):
        """ Main Function to calculate the stepsize sigma^2 and offset a^2 of a 
            random walk with noise.
        """
        with bar.ProgressBar(max_value=self.tmax-self.tmin+1) as progbar:
            for t,step in enumerate(range(self.tmin,self.tmax+1)):
                # Full trajectory s2, a2
                if not self.multi:
                    for d in range(self.ndim):
                        if self.ndim == 1:
                            z = self.z[::step] # full traj
                        else:
                            z = self.z[d,::step] # full traj
                        n = len(z) - 1
                        msd = compute_MSD_1D_via_correlation(z)[1:(self.m+1)]
                        self.a2full[t,d], self.s2full[t,d] = calc_gls(n,self.m,msd,self.d2max,self.nitmax,step)

                    a2full_3D = np.sum(self.a2full[t]) # sum across dims
                    s2full_3D = np.sum(self.s2full[t]) # sum across dims

                # segments
                n = int(self.nperseg / step) # per segment for given timestep
                # print('length N per seg = {}'.format(n))
                for s in range(self.nseg):
                    msds = np.zeros((self.ndim,self.m)) # dimensions
                    if self.multi: 
                        self.z = self.zs[s]
                        zstart = None
                        zend = None
                    else:
                        zstart = s * (self.nperseg+1)
                        zend = (s+1) * (self.nperseg+1)
                    for d in range(self.ndim):
                        if self.ndim == 1:
                            z = self.z[zstart:zend:step] # copy segment from trajectory
                        else:
                            z = self.z[d,zstart:zend:step] # copy segment from trajectory
                        msds[d] = compute_MSD_1D_via_correlation(z)[1:(self.m+1)]
                        self.a2[t,s,d], self.s2[t,s,d] = calc_gls(n,self.m,msds[d],self.d2max,self.nitmax,step)
                    msds_3D = np.sum(msds,axis=0) # --> MSD_3D per deltaT and segment
                    a2_3D = np.sum(self.a2[t,s]) # sum across dims
                    s2_3D = np.sum(self.s2[t,s]) # sum across dims

                    if self.multi:
                        self.q[t,s] = calc_q(n,self.m,a2_3D,s2_3D,msds_3D,a2_3D,s2_3D,self.ndim) # no 'full' trajectory to use
                    else:
                        self.q[t,s] = calc_q(n,self.m,a2_3D,s2_3D,msds_3D,a2full_3D,s2full_3D,self.ndim) # use a2full_3D and s2full_3D from 'full' trajectory for cinv

                a2m = np.mean(self.a2[t],axis=0) # mean across segments, per dim
                s2m = np.mean(self.s2[t],axis=0) # mean across segments, per dim

                self.s2var[t] = eval_vars(n,self.m,a2m,s2m,self.ndim,step) # a2 and s2 are mean over segments, but still per dim (and for given timestp)
            
                self.a2[t] /= step
                self.s2[t] /= step
                self.s2var[t] /= step**2
                if not self.multi:
                    self.a2full[t] /= step
                    self.s2full[t] /= step
                progbar.update(t)

    # Output and plotting
    def analysis(self,tc=10):        
        if tc % self.dt != 0:
            raise ValueError('tc must be a multiple of dt!')
        else:
            itc = int(tc/self.dt) - self.tmin
        self.Dseg = self.s2.sum(axis=2) # across dims
        self.Dseg = np.mean(self.Dseg,axis=1) / (2.*self.ndim*self.dt) # mean across segs, nm^2 / (dt * ps)
        self.Dstd = np.sqrt(self.s2var/ (2.*self.ndim*self.dt)**2) # nm^2 / (dt * ps)

        if self.multi: # no 'full' run available
            self.D = self.Dseg # nm^2 / (dt * ps)
            self.Dperdim = np.mean(self.s2,axis=1) / (2.*self.dt) # mean across segs
        else: # use full run
            self.D = self.s2full.sum(axis=1)/(2.*self.ndim*self.dt) # nm^2 / (dt * ps)
            self.Dperdim = self.s2full / (2.*self.dt)

        self.Dempstd = np.var(self.s2,axis=1) # across segments per dim
        self.Dempstd = np.sum(self.Dempstd,axis=1) # across dims
        self.Dempstd = np.sqrt(self.Dempstd) / (2.*self.ndim*self.dt)
        self.q_m = np.mean(self.q,axis=1)
        self.q_std = np.std(self.q,axis=1)

        with open('{}.dat'.format(self.fout),'w') as g:
            g.write("DIFFUSION COEFFICIENT ESTIMATE\n")
            g.write("INPUT:\n")
            g.write("Trajectory: {}\n".format(self.fz))
            g.write("Number of dimensions : {}\n".format(self.ndim))
            g.write("Min/max timestep: {}/{}\n".format(self.tmin,self.tmax))
            g.write("Number of segments: {}\n".format(self.nseg))
            g.write("Total number of trajectory data points per dim.: {}\n".format(self.n+1))
            g.write("Data points per segment and dim.: {}\n".format(self.nperseg+1))

            g.write("Your chosen diffusion coefficient at {} ps: {} nm^2/ps\n".format(tc,self.D[itc]))
            g.write("DIFFUSION COEFFICIENT OUTPUT SUMMARY:\n")
            g.write("t[ps] D[nm^2/ps] varD[nm^4/ps^2] Q\n")
            for t,step in enumerate(range(self.tmin,self.tmax)):
                g.write("{:.4g} {:.5g} {:.5g} {:.5f}\n".format(step*self.dt,self.D[t],self.Dstd[t]**2,self.q_m[t]))
            if self.ndim > 1:
                g.write("\n\DIFFUSION COEFFICIENT PER DIMENSION:\n")
                g.write("TIMESTEP Dx[nm^2/ps] Dy[nm^2/ps] ...\n")
                for t, Dt in zip( (range(self.tmin,self.tmax+1)), self.Dperdim):
                    g.write("{:.4f} {}\n".format(t, Dt))
        fig, ax = plt.subplots(2,1,figsize=(6,6),sharex=True)
        xs = np.arange(self.tmin*self.dt,(self.tmax+1)*self.dt,self.dt)
        ax[0].plot(xs,self.D,color='C0',label=r'$D$')
        # ax[0].plot(xs,Dseg,color='tab:orange', label= 'mean D segm')
        ax[0].plot(xs,self.D-self.Dstd,color='black',linestyle='dotted', label=r'$\delta \overline{D}^\mathrm{predicted}$')
        ax[0].plot(xs,self.D+self.Dstd,color='black',linestyle='dotted')
        ax[0].fill_between(xs,self.D-self.Dempstd,self.D+self.Dempstd,color='C0',alpha=0.5, label = r'$\delta \overline{D}^\mathrm{empirical}$')
        ax[0].axvline(tc,color='tab:red',linestyle='dashed')
        ax[0].set(ylabel='diff. coeff. $D$ [nm$^2$ ps$^{-1}$]')
        ax[0].set(xlim=(self.tmin*self.dt,(self.tmax+1)*self.dt))
        ax[0].ticklabel_format(style='scientific',scilimits=(-3,4))
        ax[0].legend(ncol=2)
        ax[1].plot(xs,self.q_m,color='C0')
        ax[1].fill_between(xs,self.q_m-self.q_std,self.q_m+self.q_std,color='C0',alpha=0.5)
        ax[1].axhline(0.5,linestyle='dashed',color='gray',linewidth=1.2)
        ax[1].axvline(tc,color='tab:red',linestyle='dashed')
        ax[1].set(ylabel='quality factor Q')
        ax[1].set(xlabel='time step size [ps]')
        ax[1].set(ylim=(0,1))
        fig.tight_layout(h_pad=0.1)
        fig.savefig('{}.{}'.format(self.fout,self.imgfmt),dpi=300)

    def finite_size_correction(self,T=300,eta=None,L=None,boxtype='cubic',tc=10):
        """ T in Kelvin, eta in Pa*s, L in nm"""
        if tc % self.dt != 0:
            raise ValueError('tc must be a multiple of dt!')
        else:
            itc = int(tc/self.dt) - self.tmin
        if self.ndim != 3:
            raise ValueError("Currently only 3D correction implemented")
        if L == None:
            raise ValueError("Box edge length L missing.")
        if eta == None:
            raise ValueError("Viscosity eta missing.")
        if boxtype != 'cubic':
            raise ValueError("Sorry, correction only implemented for cubic simulation boxes")

        xi = 2.837297
        kbT = T * 1.380649e-23 # J
        self.Dcor = self.D + kbT * xi * 1e15 / (6. * np.pi * eta * L) # nm^2 / ps
        print("Finite-size corrected diffusion coefficient D_t for timestep {} ps: {:.4g} nm^2/ps with standard dev. {:.4g} nm^2/ps".format(tc,self.Dcor[itc],self.Dstd[itc]))
