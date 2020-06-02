# Diffusion Coefficient Fitting

Python class library to determine the diffusion coefficient of a time series using a Generalized Least Squares (GLS) minimization procedure, which accounts for the correlation of MSD data.

Please read and cite the reference: J. Bullerjahn, S. v. Bülow, G. Hummer, Optimal estimates of diffusion coeffcients from molecular dynamics
simulations, Journal of Chemical Physics XXX, YYYYY (2020).

The input trajectory is analyzed as a whole and split into segments. For each segment, a quality factor Q is computed, indicating how well the trajectory fits a model of random diffusion with noise. The analysis is done for different time steps $\Delta t_n$ of the trajectory. Given the quality factor analysis, the user decides on a time step/diffusion coefficient pair to use.

# Basic Example
```
import Dfit
res = Dfit.Dcov(m=20,fz='mytrajectory.dat',tmin=1,tmax=100,nseg=150)
res.run_Dfit()
res.analysis(tc=10)
res.finite_size_correction()
```

# Input for Dfit.Dcov()

Required:
* fz (str): Filename of input trajectory. Format: Center-of-mass position x [y z ...] in nm. Each row corresponds to a timestep indicated in the optional argument `dt`. The number of columns determines the number of dimensions. Use no header. Separate columns by whitespace, no comma.

*(NOTE: Alternatively, you can provide a list of input trajectories. The code will then treat these lists as copies of the same molecule in the same simulation [e.g. several water molecules, or several copies of the same protein in a dense protein solution, or several DOPC lipids...]. The individual trajectories will then not be segmented; instead, each individual trajectory will be treated as one segment of a long simulation. The resulting diffusion coefficient is then the mean across all segments.)*

Optional:
* dt (float): Timestep [in ps] (default: 1.0).
* m (int): Number of MSD values to consider (default: 20).
* tmin (int): Minimum timestep (default: 1).
* tmax (int): Maximum timestep (default: 100).
* d2max (float): Convergence criterion for GLS iteration (default: 1e-10).
* nitmax (int): Maximum number of iterations in GLS procedure (default: 100).
* nseg (int): Number of segments (default: N / (100*tmax)).
* fout (str): Name for output files w/o extension (default 'D_analysis').
* imgfmt (str): Output format for plot. Choose 'pdf' or 'png' (default: 'pdf').

Run res.run_Dfit() without additional arguments (everything is stored in `self`).

# Output

Files:
* D_analysis.dat: Summary of output, including diffusion coefficient and Q-factor analysis.
* D_analysis.pdf: (or .png) Output plots. Use this to assess which diffusion coefficient to select. Refer to Figs. 2 and 4 in the paper.

Stored values in the Dfit class instance:
* res.D: Optimal diffusion coefficient estimates per timestep
* res.Dstd: Estimated standard deviation of res.D per timestep
* res.Dempstd: Empirical standard deviation of res.D per timestep (should be close to res.Dstd if the diffusive model is appropriate)
* res.q_m: Mean quality factor per timestep
* res.q_std: Standard deviation of quality factor per timestep

# Additional input for method analysis():

* tc (int/float): 'Timestep chosen' [in ps] for the diffusion coefficient. Must be a multiple of dt.

The analysis can be repeated with different values of tc. A red vertical line indicates the chosen timestep in the plot. Repeat until you are content with the chosen timestep

# Additional input for finite_size_correction():
* tc (int/float): Use tc determined in the previous analysis step
* eta (float): Viscosity in units of Pa*s
* L (float): Edge length of cubic (!) simulation box in nm
* T (float): Temperature in Kelvin
* boxtype (str): Only 'cubic' currently allowed.
