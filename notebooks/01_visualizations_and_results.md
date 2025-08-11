# Recap and Introduction

This notebook compares the performance of two methods of computing vorticity, a measure of rotation in ocean currents. We adapted a novel method called Lagrangian Gradient Regression (LGR; Harms et al. 2023) and our goal is to compare it to a more established ocean model called ROMS.

We will start with the visualizations I created to make the results clear to a broad audience. Later parts of the notebook will provide code for the plots, then expand on the technical details of how we computed vorticity and the statistical metrics used in the plots.

# Visualizing Vorticity - Plot Design and Experiment Results

Code for these plots will follow in the next section.

### Plot 1: Waves

![Trajectory Plot](../images/quiver.png)

Our simulations took place in an idealized coastal environment, where consistant wind-driven waves create complex flow patterns.

Two features of this plot are designed to stand out against the dark color scheme: The land, in white, and the breaking wave motion, in dark red. This ensures that the viewer can quickly center themselves on a simple, broad picture: We are simulating waves breaking against a coastline.

The vectors on the plot show the direction and speed of the complex currents which result from the waves approaching the coast. The patterns are distinctly circular, rotating, intuitively connecting to vorticity.

Finally, the color gradient of the plot shows the speed, providing quantifyable context to the magnitudes of the flow vectors. The color scheme is purposely meant to be unintrusive, since other aspects of the plot are likely going to be more important to most viewers.

### Plot 2: Trajectories

![Trajectory Plot](../images/trajectories.png)

As discussed in the previous notebook, LGR computes vorticity from the trajectories of floating test particles in the ocean. We will be using simulated particle trajectories to perform our test of LGR.

This plot demonstrates what would happen if a particle (such as a buoy) was thrown into the currents shown in the last plot. Starting at the initial seeding positions (the red dots), the trajectories follow by simply tracing a path along the vectors in the last plot. This is actually a remarkably good analogue for how this computation actually works, numerically integrating a system of differential equations.

This plot also again uses a subtle color scheme to introduce information about wave height: The initial wind-driven waves are 3 meters high. This is a very windy day. 

### Plot 3: Vorticity Computation Comparison

![LGR vs. ROMS Plot](../images/ROMS_LGR.png)

This plot shows the results of the LGR vorticity computation alongside the output from the established ocean model ROMS (see the Data Analysis section below for a more technical discussion about how this data was generated).

The plots' bimodal color scheme helps show that the magnitude of vorticity represents rotation speed and that the sign (positive or negative) of vorticity represents rotation direction.

The similarities of the two plots, especially inside the dashed line signifying the edge of where particles were seeded, support our conclusion that LGR has potential as a vorticity computation method.

### Plot 4: LGR Error

![LGR vs. ROMS Plot](../images/errorPlots.png)

To more rigorously define the effectiveness of LGR relative to ROMS, we performed a root mean squared error (rmse) analysis (described more in the Vorticity Computation and Statistical Analysis sections below). For the sake of these tests, we assumed that the highly tested and frequently used ROMS model was perfectly accurate, and compared our LGR outputs with a vareity of initial conditions to the ROMS output.

This plot shows the results of those tests. In the first plot, the number of seeded particles was kept at 1200, but the number of depth contours, or isobaths, on which particles were seeded was varied (the larger the distance between isobaths, the fewer isobaths were used). The relative error was minimized when the distance between isobaths was just under a meter, leading to the spacing seen in the other plots.

The second plot shows that as time advances (from the lowest error initial spacing), the particles leave their starting positions and LGR becomes less effective, stabilizing at a high error after about 15 minutes.

### Animation 1: Particle Motion

![Particle Animation](../images/hour.gif)

These analyses put together confirmed that LGR can be an effective technique to compute vorticity that, at least under certain conditions can perform as well as ROMS. However, the quick increase in error as time progresses raises concerns about the applicability of this approach in the field.

This animation aims to diagnose the causes of these discrepencies. We believe that the error increases over time because, as this animation shows, the particles quickly form clusters, leaving other areas blank. After about 15 minutes, the waters near the central coastlines (with the highest vorticity values) are flushed clean, corresponding to the breakdown in accuracy. These high speeds can also be seen graphically in Plot 1.

This implies that LGR can be best used in places with slower water motion, where the test particles will maintain optimal coverage longer.

# Visualizing Vorticity - Plot Code Walkthrough

This section will provide a walkthough of the Python code I wrote to create these visualizations. We will use the following libraries:

```python
import xarray as xr
import numpy as np
import zarr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.colors as colors
import cmasher
import pandas as pd
import cmocean
from scipy import stats
from scipy.ndimage import median_filter, uniform_filter
```

If you have read the previous notebook (01_generate_particle_data.md), ds will now be called dsCDF to differentiate it from the trajectory data, dsTRAJ.

dsCDF is a multidimensional dataset in the netCDF format which contains information about our simulated coastline's water's depth and motion.

```python
# trajectory data output from the previous notebook
dsTRAJ = xr.open_zarr('/Users/jordan/Documents/CICOES/data/posterData/1200p_15c_dt15.zarr')

# seconds per step, the number of seconds between each observation in the trajectory data
sps = 30 

# called ds in the previous notebook
dsCDF = xr.load_dataset('/Users/jordan/Documents/CICOES/data/cape_large_00.nc', decode_times = False)

```

### Plot 1: Waves

We extract relevant data from dsCDF; for the matplotlib pcolormesh plot of speed, we will need x and y coordinates, alongside velocities in the x and y directions, denoted with U and V in this dataset. These variables use the psi positional coordinates.

```python
# unfiltered versions of velocity/coordinates for pcolormesh
Xfull = dsCDF.x_psi
Yfull = dsCDF.y_psi
Ufull = dsCDF.ubar_lagrangian_psi
Vfull = dsCDF.vbar_lagrangian_psi

# create a multidimensional array of speed at all points
flowmag = np.sqrt(Ufull**2 + Vfull**2) 
```

For the vectors, we use a matplotlib quiverplot. It needs similar data, filtered to not have too many vectors and to keep the top of the plot clear.

```python
# Filter rates
cffqu = 8
cffqv = 8

# filtered versions of velocity/coordinates for quiverplot
Xq = dsCDF.x_psi[::cffqu, ::cffqv]
Yq = dsCDF.y_psi[::cffqu, ::cffqv]
Uq = dsCDF.ubar_lagrangian_psi[::cffqu, ::cffqv]
Vq = dsCDF.vbar_lagrangian_psi[::cffqu, ::cffqv]

# Turn all values above 700 to NaN so the top of the plot is clear
Yq = np.where(Yq > 700, np.nan, Yq)

# scaled vector lengths for quiver plot
Unorm = Uq / (Uq**2 + Vq**2)**0.25
Vnorm = Vq / (Uq**2 + Vq**2)**0.25
```
The dark red arrows are drawn using another quiverplot, requiring similar parameters once more.

```python
# define parameters for the wave indicator arrows,
# spaced to avoid arrows spawining directly at the edge of the plot
Xwave = np.arange(-980,980 + 0.001,490)
Ywave = [800] * len(Xwave)
Uwave = [0] * len(Xwave)
Vwave = [-100] * len(Xwave)
```

As a final step before starting the plot, I define a custom colormap for the meshgrid.

```python
cmgray_middle = cmasher.get_sub_cmap(cmocean.cm.gray_r, 0, 0.6)
```

Then, using standard matplotlib techniques, I complete the plot.

```python
fig, ax = plt.subplots(figsize = (14,6.5), dpi = 1200, constrained_layout = True)

# draws the land based on a filter included in dsCDF
ax.pcolormesh(dsCDF.x_psi, dsCDF.y_psi, np.ma.masked_where(dsCDF.mask_psi == 1, dsCDF.mask_psi), 
              cmap = colors.ListedColormap(['#FAF6EB', '#ffffff00']))

# flow magnitude pcolormesh drawn, 
# colors assigned on a log scale to better represent the data
vel = ax.pcolormesh(Xfull, Yfull, flowmag, cmap = cmgray_middle, norm = colors.LogNorm(vmin = 1e-3, vmax = 1))

# speed vectors drawn
ax.quiver(Xq,Yq,Unorm,Vnorm, scale = 20)

# wave incoming quiver
ax.quiver(Xwave,Ywave,Uwave,Vwave, color = 'darkred', headaxislength = 4, headlength = 4)

ax.set_title('Incoming Breaking Waves', size = 28, pad = 10)
ax.set_xlabel(r'$x$ (m)', size = 24)
ax.set_ylabel(r'$y$ (m)', size = 24)

ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)

ax.set_ylim(0,800)
ax.set_xlim(-1050,1050)
ax.set_aspect("equal")

# legend for speed pcolormesh
cbar = fig.colorbar(vel, orientation = "vertical", fraction = 0.05, aspect = 20, shrink = 0.652, pad = 0.01) # pad = 0.20
cbar.set_label(label='Speed (m/s)', size=28)
cbar.ax.tick_params(labelsize = 24)

plt.savefig('plots/poster_video/quiver.png')
plt.close(fig)
```

### Plot 2: Trajectories

This plot begins simialrly to the last one, since we are again using a matplotlib meshgrid. The wave height data (stored under the variable 'Hwave') uses the rho position coordinates.

```python
# get position coordinates
x_rho = dsCDF.x_rho[0,:].values 
y_rho = dsCDF.y_rho[:,0].values
Xrho, Yrho = np.meshgrid(x_rho, y_rho)

# create the colormap 
cmice_middle = cmasher.get_sub_cmap(cmocean.cm.ice_r, 0.1, 0.7)

# begin figure
fig, ax = plt.subplots(figsize = (14,6.5), dpi = 1200, constrained_layout = True)

# draws the land
ax.pcolormesh(dsCDF.x_psi, dsCDF.y_psi, np.ma.masked_where(dsCDF.mask_psi == 1, dsCDF.mask_psi), 
              cmap = colors.ListedColormap(['#FAF6EB', '#ffffff00']))

# wave height contour
wav = ax.pcolormesh(Xrho, Yrho, dsCDF['Hwave'], cmap = cmice_middle)
```

In order to draw the trajectories, we need to directly use the trajectory data from the last notebook. This plot uses 1024 particles instead of the usual 1200 for a clearer picture.

```python
chosenparts = np.arange(0, 1025, 1)

# plot each particle's starting position and trajectories
for i in chosenparts :
    ax.scatter(dsTRAJ['lon'].values[i][0], dsTRAJ['lat'].values[i][0], c = 'red', s = 1.5)
    ax.plot(dsTRAJ['lon'].values[i], dsTRAJ['lat'].values[i], c = 'k', linewidth = 0.3)
```
After this point, the plot uses standard matplotlib code.

```python
ax.set_xlabel(r'$x$ (m)', size = 24)
ax.set_ylabel(r'$y$ (m)', size = 24)

ax.xaxis.set_tick_params(labelsize=24)
ax.yaxis.set_tick_params(labelsize=24)

ax.set_ylim(0,800)
ax.set_xlim(-1050,1050)
ax.set_aspect("equal")

# create meshgrid legend
cbar = fig.colorbar(wav, orientation = "vertical", fraction = 0.05, aspect = 20, shrink = 0.669, pad = 0.01) # pad = 0.20
cbar.set_label(label='Wave Height (m)', size=28)
cbar.ax.tick_params(labelsize = 24)

plt.savefig('plots/poster_video/trajectories.png')
plt.close(fig)
```

### Plot 3: Vorticity Computation Comparison

As this plot has two panes, we use plt.subplot, creating each plot on separate axes.

```python
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, constrained_layout = True, figsize = (14,14), dpi = 1200) 
```

The top panel, showing ROMS vorticity, is a standard matplotlib pcolormesh. The data variable romvortmean's creation appears in the Vorticity Computation section below.

```python
romvmeanplt = ax1.pcolormesh(X, Y, romvortmean, cmap = cmocean.cm.balance, 
                        norm=colors.SymLogNorm(linthresh=0.01, base=2)) # can add vmin and vmax in here

# draws the land
ax1.pcolormesh(dsCDF.x_psi, dsCDF.y_psi, np.ma.masked_where(dsCDF.mask_psi == 1, dsCDF.mask_psi), 
               cmap = colors.ListedColormap(['#FAF6EB', '#ffffff00']))

# the outer boundary of the particle seeding zone is a distinct dashed line
ax1.contour(Xrho, Yrho, dsCDF['h'], levels = [10], colors = 'black', linestyles = 'dashed')

# axis  1 atributes
ax1.set_title('ROMS Vorticity', size = 32)
#ax1.set_xlabel(r'$x$ (m)')
ax1.set_ylabel(r'$y$ (m)', size = 24)

ax1.yaxis.set_tick_params(labelsize=24)

ax1.set_ylim(0,800)
ax1.set_xlim(-1050,1050)
ax1.set_aspect("equal")
```

The two plots use a shared colorbar, which uses a symetric logarithmic scaling with even bounds on each side to address the negative and positive vorticities not having precisely the same maximum absolute magnitude. 

The data variable vort_nofilter contains the LGR vorticity data, and will again be derived in the Vorticity Computation section.

```python
vmin, vmax = -np.nanmax(vort), np.nanmax(vort)

# a norm defining how to assign colors to numbers
norm = colors.SymLogNorm(linthresh = 0.01, base = 2, vmin = vmin, vmax = vmax)

# LGR vorticity pcolormesh
vplt = ax2.pcolormesh(X, Y, vort_nofilter, cmap = cmocean.cm.balance, norm = norm)  

# draws the land
ax2.pcolormesh(dsCDF.x_psi, dsCDF.y_psi, np.ma.masked_where(dsCDF.mask_psi == 1, dsCDF.mask_psi), 
               cmap = colors.ListedColormap(['#FAF6EB', '#ffffff00']), zorder = 10)

# draws slightly faint vorticity contour lines
levels = np.linspace(vmin, vmax, 25)
ax2.contour(X, Y, vort_nofilter, levels = levels, linewidths = 0.25, alpha = 0.75, colors = "k") 

# the outer boundary of the particle seeding zone is a distinct dashed line
ax2.contour(Xrho, Yrho, dsCDF['h'], levels = [10], colors = 'black', linestyles = 'dashed')
```
The LGR plot also has text showing the timestep and the  rmse (root mean squared error, also called the mean L2 norm) of the configuration (see the Statistical Analysis section below).

```python
# displays the simulation time - this plot shows the first frame, time 0.
ax2.text(1000,600,'time: {} min'.format(tstep*sps/60), size = 24, horizontalalignment='right', verticalalignment='center')

# displays the rmse/mean L2 norm
ax2.text(-1000,600, 'rmse: {}'.format(meanl2norm), size = 24, horizontalalignment='left', verticalalignment='center')
```

The rest of the plot uses standard matplotlib code, including the creation of the (shared) colorbar.

```python
# axis 2 atributes
ax2.set_title('LGR Vorticity', size = 32)
ax2.set_xlabel(r'$x$ (m)', size = 24)
ax2.set_ylabel(r'$y$ (m)', size = 24)

ax2.xaxis.set_tick_params(labelsize=24)
ax2.yaxis.set_tick_params(labelsize=24)

ax2.set_ylim(0,800)
ax2.set_xlim(-1050,1050)
ax2.set_aspect("equal")


# remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0)


# colorbar - pad is distance below to place it
cbar = fig.colorbar(vplt, orientation = "horizontal", fraction = 0.05, aspect = 30, ax = (ax1,ax2), shrink = 1) # pad = 0.20
cbar.set_label(label='Vorticity (1/s)', size=28)
cbar.ax.tick_params(labelsize = 24)

plt.savefig('plots/poster_video/ROMS_LGR.png')
plt.close()
```

### Plot 4: LGR Error

This plot itself uses standard matplotlib subplot code. Derivations of the data variables involved can be found in the Vorticity Computation and Statistical Analysis sections below.

```python
# subplots using the same y axis scale
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharey = True, figsize = (14,13.75), dpi = 1200)

# make a scatter plot, connecting data with lines
ax1.scatter(cont_space, rmse, c = 'k', s = 200)
ax1.plot(cont_space, rmse, c = 'k', linewidth = 5)

# remove the top and right parts of the frame
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.set_xlabel('Isobath Spacing (m)', size = 24)
ax1.set_ylabel('Error', size = 24)

# set minor tickmarks for the top plot x-axis
ax1.set_xticks(np.linspace(0,5,26), minor=True)

# set the size of the major tick labels
ax1.xaxis.set_tick_params(labelsize=24)
ax1.yaxis.set_tick_params(labelsize=24)


# make a scatter plot, connecting data with lines
ax2.scatter(times, rmse_t, c = 'k', s = 60)
ax2.plot(times, rmse_t, c = 'k', linewidth = 4)

# remove the top and right parts of the frame
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.set_xlabel('Time (min)', size = 24)
ax2.set_ylabel('Error', size = 24)

# set the tickmarks for the plot - the bottom plot x-axis
ax2.set_xticks(np.arange(0,60,2), minor=True)
ax2.set_yticks(np.arange(0.007,0.019,0.002), minor=True)

# set the size of the major tick labels
ax2.xaxis.set_tick_params(labelsize=24)
ax2.yaxis.set_tick_params(labelsize=24)


# remove space between subplots
plt.subplots_adjust(wspace=0, hspace=0.3)

plt.savefig('plots/poster_video/errorPlots.pdf')
plt.close(fig)
```

### Animation 1: Particle Motion

The animation is made by looping through a series of plots and saving the resulting files, then using FFmpeg command-line code to turn them into an animated gif.

The plots are very similar to the Seeding Diagram plot highlighted in the previous notebook, with the $t=0$ LGR plot from Plot 3 superimposed underneath.

All plots use the same Plot 3 vorticity setup.

```python
vmin, vmax = -max(df['vorticity'].values[0]), max(df['vorticity'].values[0])

# define the color norm for all plots
norm = colors.SymLogNorm(linthresh = 0.01, base = 2, vmin = vmin, vmax = vmax)

# levels for the contour plots
levels = np.linspace(vmin, vmax, 25)

x_rho = dsCDF.x_rho[0,:].values 
y_rho = dsCDF.y_rho[:,0].values
Xrho, Yrho = np.meshgrid(x_rho, y_rho)
```

Now, we enter the for loop. The dataframe df is generated in the LGR vorticity computation code described in the Vorticity Computation section below. Here, it is used to obtain trajectory information in place of dsTRAJ.

```python
for i in range(len(df['positions'])) :

    fig, ax = plt.subplots(figsize = (7,5), dpi = 300) 

    # LGR vorticity pcolormesh like Plot 3
    vplt = ax.pcolormesh(X, Y, vort_nofilter, cmap = cmocean.cm.balance, norm = norm)  # old cmap ReBu_r?

    # draws the land
    ax.pcolormesh(dsCDF.x_psi, dsCDF.y_psi, np.ma.masked_where(dsCDF.mask_psi == 1, dsCDF.mask_psi), 
                  cmap = colors.ListedColormap(['#FAF6EB', '#ffffff00']), zorder = 10)
    
    # vorticity contour plot like Plot 3
    ax.contour(X, Y, vort_nofilter, levels = levels, linewidths = 0.25, alpha = 0.5, colors = "k")


    # adds a scatterplot of the positions of particles at the timestep we are plotting
    ax.scatter(df.loc[i].positions[:,0], df.loc[i].positions[:,1], s = 0.15, color = "k", zorder = 15)  # s = 0.25
    
    # t
    ax.text(1000,900,'t = {} min'.format(i*sps/60),
            horizontalalignment='right',
            verticalalignment='center')
    
    
    # particle counter
    ax.text(-1000,900, '{} particles'.format(np.count_nonzero(~np.isnan(df['vorticity'].values[i]))),
            horizontalalignment='left',
            verticalalignment='center')

    ax.set_ylim(0,1000)
    ax.set_xlim(-1050,1050)
    ax.set_aspect("equal")
    
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    ax.set_title('Vorticity and Particle Positions', size = 15)

    # colorbar - pad is distance below to place it
    cbar = fig.colorbar(vplt, orientation = "horizontal", fraction = 0.05, aspect = 30, pad = .20)
    cbar.set_label(label='Vorticity', size=15)
    
    # this pads the strings with zeros so that everything is placed in order for easy gif making
    plt.savefig('plots/poster_video/animation/poster/image-'+str(i).zfill(4)+'.png')
    plt.close(fig)
```

At this point, we have a folder with all of the plot images in it. The following FFmpeg command-line code generates the final animated gif:

```bash
# from https://stackoverflow.com/questions/58832085/colors-messed-up-distorted-when-making-a-gif-from-png-files-using-ffmpeg
# this generates a palette to prevent a weird yellow glow on the final product
ffmpeg -i image-%04d.png -vf palettegen palette.png

# from https://www.bannerbear.com/blog/how-to-make-gifs-from-images-using-ffmpeg/#selecting-a-range-of-files
# this line creates the gif 
# -framerate -> frames per second - 10 works well
# -i takes images of the form image-0001 to image-9999 in order, zero padding as done in the code block above
# -i again to set the palette from the last step
# finally, the file is saved in the folder with the plot images as hour.gif
ffmpeg -framerate 10 -i image-%04d.png -i palette.png -lavfi paletteuse hour.gif
```

# Vorticity Computation - Code Walkthough

For completeness, this section will describe how the trajectory data in the previous notebook (01_generate_particle_data.md) is used to compute and compile the vorticity data used in the above plots.

First, the ROMS vorticiy is actually contained in dsCDF already. It is computed directly from our simulated water speed. 

Mathematically, vorticity is defined to be the curl of a flow field: $ \omega := \nabla V $ 

In other words, vorticity mathematically quantifies the degree to which a vector field (like in Plot 1) circulates around any given point. Since we have a full flow vector field already in our simulated environment, ROMS easily performs the needed computation, and we can just read in the output.

```python
romvort = dsCDF['ω_bar'].values
```

The LGR vorticity computation relies on a slightly altered version of the LGR model developed by Harms et al., the original proprietary code for which I cannot share here. 

We import their model code:

```python
from LGR_altered.lgr import *
from LGR_altered.jacobian import *
from LGR_altered.classes import *
from LGR_altered.plotting import *
```

Before the computations run, there is some setup required. The first step is to define our regression method. For this analysis, we use radial Gaussian regression.

```python
# Generate the regression function
regfun = setRegressionFunction(kernel=reg_type, lam=lam, sig=sigma)
```
Next, the data from dsTRAJ (the trajectory data from the previous notebook) is used to create an array called particleList, which catalogues each particle as a SimpleParticle class object with position and time data as atributes. The other parameter kNN determines how many neighboring particles to use in the regression. We set kNN = 5.

We also obtain a count of particles.

```python
# Generate a data frame
df = generateDF(particleList, kNN)
n_particles = len(df['indices'][0])
```

Now, we can run the full model. 

```python
# Perform the regressions
calcJacobianAndVelGrad(df, regfun=regfun)

# Compute the metrics on each particle trajectory
# The primary metric of interest is LGR vorticity
computeMetrics(df, t, metric_list=metrics)

# drop the last row as it is prone to errors
df = df[:-1]
```
In a rough sense, what these functions do is record the direction of motion of every particle at each time step relative to the five nearest other particles, weighted by distance, and from these rates of change compute vorticity, the amout of relative rotation. 

Mathematically, at every point, our goal is to compute vorticity in the same way as with ROMS, using the formula $ \omega := \nabla V $. However, starting from just trajectory data, we don't have the necessary vector field V. Instead, we approximate it via a complex process using Gaussian-weighted regression. As time progresses one small step, we track the change in distances between each particle and its 5 nearest neighbors. Regression then gives a matrix which best transforms the old positions to the new ones, an approximate flow matrix. At any given timestep, a composition of these flow matrices gives an approximation of $\nabla V$, allowing vorticity to be approximated.

Continuing to set up the data for statistical analysis, the LGR vorticity data needs to be interpolated back onto our simulation's gridded coordinates, used by ROMS.

```python
# this generates the meshgrid from the x and y values of the LGR model grid
# we make the grid sparcer for efficiency
xvec = dsCDF.x_psi[0,:].values 
yvec = dsCDF.y_psi[:,0].values 

gridvectors = [xvec, yvec]

# choose interpolation method based on the number of particles
if n_particles < 2000:  
    generateFields(df, gridvectors, approach='rbf', method='multiquadric', smooth = smooth)
    interpstr = 'rbf_mq'
else:
    generateFields(df, gridvectors, approach='interp', method='cubic')
    interpstr = 'int3'
```

The critical parameter which I added to the generateFields function was the smoothing parameter. When interpolating data to fit a grid, especially when the computed data is concentrated only around where the particles currently are, requires the computer to "guess" at how to fill in the empty regions. Adding the smoothing parameter greatly reduces extranious values in the corners of the plot.

We can then extract the LGR vorticity, called vort in this code. A raw unfiltered copy is made, vort_nofilter, which is the data variable in the LGR panel in Plot 3.

```python
### our computed quantities
# .loc selects a timestep, and the scalarfields column. Each element is a dictionary, so we pull out the one we want
# Each dictionary contains a 2d array array, horizontal values in rows, vertical values in columns, used for plotting
vort = np.squeeze(df.loc[tstep, 'ScalarFields']['vort'])
vort_nofilter = np.copy(vort)
```

Finally, to make the ROMS vorticity data be more similar in form to the interpolated LGR vorticity data, we "smooth" it as well, by replacing each value on the plot with the local mean of the surrounding 5 by 5 box of values, along with some operations to handle some edge cases NaNs.

```python
# where romvort is nan, plug in 0, otherwise keep the original value
# this will prevent NaNs from interering with the local means
romvort_nonan = np.where( np.isnan(romvort), 0, romvort)

# a grid with NaNs as NaN, and 0s elsewhere
romvort_nans = np.where(np.isnan(romvort), np.nan, 0)

# take the local mean at every point
romvortmean = uniform_filter(romvort_nonan, size = 5, mode = 'nearest')

# create a version of romvortmean where every nan value is put back in place after the computation
# number + NaN = NaN
romvortmean_nans = romvortmean + romvort_nans
```

The romvortmean varaible is the data variable in the ROMS panel in Plot 3.

# Statistical Analysis - Code Walkthough

We are now in a position to explain the details behind the statistical analyses in Plot 4.

The errors themselves are computed via the mean L2 norm, or root mean squared error (rmse). The ROMS data is treated as the theoretical, correct, value while the LGR data is considered experimental.

The equation for the computation is as follows:

$ \text{rmse} = \sum^n_{i=0} \sqrt { \frac{(\text{LGR}[i] - \text{ROMS}[i])^2} {n} }$

We take the differences, square them, take the root, and take the mean (by dividing by n, the total number of points), hence the name root mean squared error.

```python
errorMesh = (vort - romvortmean_nans)**2
meanl2norm = round( np.sqrt(np.nansum(errorMesh)/ np.count_nonzero(~np.isnan(errorMesh))), 5) 
```

This code performs this computation efficiently by creating a mesh with the error at every point then summing the full mesh and dividing by the count. 

For a variety of configurations, I performed this computation, as well as a similar computation of correlation, and recorded the results in a csv.

```python
dsError = pd.read_csv('/Users/jordan/Library/.../CICOES/data/errorData.csv')

# filter out negative correlation values and the ncont = 600 case
ds_poscorr = dsError[(dsError['corr'] > 0) & (dsError['ncont'] < 600)] 

# extract the columns of interest
ncont = ds_poscorr[['ncont']].to_numpy().T[0]
rmse = ds_poscorr[['rmse']].to_numpy().T[0]
corr = ds_poscorr[['corr']].to_numpy().T[0]
```

The final task requried to create Plot 4 was to create an error time series for the lower panel.

```python
times = []
rmse_t = []
corr_t = []

# loop over the total number of time steps
for i in range(len(df['positions'])) :
    
    # sometimes the last few datapoints are flawed, we want to exit the loop without an error when this happens
    try :
        # .loc selects a timestep, and the scalarfields column. Each element is a dictionary, so we pull out the one we want
        # Each dictionary contains a 2d array array, horizontal values in rows, vertical values in columns, used for plotting
        vorti = np.squeeze(df.loc[i, 'ScalarFields']['vort'])
        vorti[ dsCDF['h_psi'][:,::cff] >= 10 ] = np.nan
    except :
        break
    
    vortveci = vorti.flatten()
    romvortveci = romvortmean_nans.flatten()
    

    errorMeshi = (vorti - romvortmean_nans)**2
    meanl2normi = round( np.sqrt(np.nansum(errorMeshi)/ np.count_nonzero(~np.isnan(errorMeshi))), 5) 
    
    maski = ~np.isnan(romvortveci) & ~np.isnan(vortveci)
    
    corri = round(stats.linregress(romvortveci[maski], vortveci[maski]).rvalue, 3)

    times.append(i*sps/60)
    rmse_t.append(meanl2normi)
    corr_t.append(corri)
```

We have now created all data variables contained in Plot 4.