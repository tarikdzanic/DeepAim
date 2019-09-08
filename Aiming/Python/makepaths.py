import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter
import math
import generatesplines as gs
import random



# Generates smooth exponential ramping from 0 to 1 to 0 for deviations from spline. 
# Appropriate values for ramp:  rampin = 0.0 and rampout = 1.0 (instant ramping with no lag and then ramp to 0 at end)
# Appropriate values for width: 0.01 - 0.03
def getRampCoeff(t, rampin, rampout, rampinwidth, rampoutwidth):
	if (rampin < 0.0 or rampin > 1.0 or rampout < 0.0 or rampout > 1.0):
		raise ValueError('Rampin/out values must be between 0.0 and 1.0')

	x1 = t - rampin
	x2 = rampout - t

	# Values before and after rampin/rampout are zero. Otherwise do exponential damping
	if (x1 > 0.0):
		c1 = np.exp(-1.0/(x1/rampinwidth))
	else:
		c1 = 0.0

	if (x2 > 0.0):
		c2 = np.exp(-1.0/(x2/rampoutwidth))
	else:
		c2 = 0.0

	return(c1*c2)

# Plots a spline given the nodes of the spline and the coefficients of the parametric representation of the curves (xc, yc) and # of points to plot with
# xc, yc represents x(t) = xc[0] + xc[1]*t + xc[2]*t^2 + ... for t c[0.0, 1.0]
# Oscillations adds sinusoidal oscillations to the path normal dicated by frequency list (radians) and amplitude list (pixels)
# Ramp adds exponential damping to oscillations near beginning and end of path

def getSplinePoints(nodelist, xc, yc, npoints, show = False, oscillations = False, freqlist = [], amplist = [], ramp = False, rampin = 0, 
					rampout = 0, rampinwidth = 0, rampoutwidth = 0, noise = False, noiseamplitude = 0):
	t = np.linspace(0,1,npoints)
	x = []
	y = []

	# Calculate x,y at each t
	for i in range(0,len(t)):
		curt = t[i]
		xsum = 0
		ysum = 0

		for j in range(0,len(xc)):
			xsum += xc[j]*curt**j
			ysum += yc[j]*curt**j
		
		# Generate sinusoidal disturbances to path
		if (oscillations):

			# Get path tangent and normal
			tanx = 0
			tany = 0

			# Take derivative of spline to find tangent
			for j in range (1,len(xc)):
				tanx += j*xc[j]*curt**(j)
				tanx += j*yc[j]*curt**(j)
			# Normalize and find normals
			tanx = tanx/np.sqrt(tanx**2 + tany**2)
			tany = tany/np.sqrt(tanx**2 + tany**2)

			normx = -tany
			normy = tanx

			# Generate disturbance as sum of sines
			wave = 0
			for k in range(0,len(freqlist)):
				wave += amplist[k]*math.sin(curt*freqlist[k])

			# Add exponential ramping to disturbances
			if (ramp):
				rampcoeff = getRampCoeff(curt, rampin, rampout, rampinwidth, rampoutwidth)
				wave *= rampcoeff

			# Get deviations to x,y position
			devx = wave*normx
			devy = wave*normy

			xsum += devx
			ysum += devy

		if (noise):
			devx = random.random()*noiseamplitude
			devy = random.random()*noiseamplitude
			xsum += devx
			ysum += devy

		x.append(xsum)
		y.append(ysum)

	if show:
		# Plot spline node locations
		# Make all nodes blue except start node (green) and end node (red)
		colors     = ['bo']*len(nodelist)
		colors[0]  = 'go'
		colors[-1] = 'ro'
		for i in range(0,len(nodelist)):
			node = nodelist[i]
			plt.plot(node.x, node.y, colors[i])
		plt.plot(x,y, 'r-')

	return([x,y])


def getSplineCoeffs(xstart, ystart, xtarget, ytarget, vx, vy, ax, ay, gradscale):
	startnode = gs.Node()
	startnode.setPos(xstart, ystart)

	startnode.setDeriv(1, [vx*gradscale[0], vy*gradscale[0]])
	startnode.setDeriv(2, [ax*gradscale[1], ay*gradscale[1]])

	endnode = gs.Node()
	endnode.setPos(xtarget, ytarget)

	# Get spline polynomial coefficients for x,y
	[spline_xcoeffs, spline_ycoeffs] = gs.makeSpline([startnode, endnode])
	return([spline_xcoeffs, spline_ycoeffs])

def makePath(npoints, xstart, ystart, xtarget, ytarget, vx, vy, ax, ay, gradscale):
	startnode = gs.Node()
	startnode.setPos(xstart, ystart)
	endnode = gs.Node()
	endnode.setPos(xtarget, ytarget)
	[spline_xcoeffs, spline_ycoeffs] = getSplineCoeffs(xstart, ystart, xtarget, ytarget, vx, vy, ax, ay, gradescale)

	[xspline, yspline] = getSplinePoints([startnode, endnode], spline_xcoeffs, spline_ycoeffs, npoints)

	return ([xspline, yspline])



