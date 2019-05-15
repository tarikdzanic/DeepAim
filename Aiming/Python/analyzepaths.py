import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter
import math
import generatesplines as gs
import random

# Number of points to smooth over for discrete derivative terms (velocity, acceleration, jerk, etc.)
smoothwindow = 51

# Path of pickled mouse movement data from trackmovement.py
datapath = 'MouseMovementData_SessID05893'

# Extract and format data from pickled mouse path files from trackmovement.py. Index corresponds to run # of test (must be > 0 since 0 is reserved for header line)
def extractData(data, index):
	if (index < 1 or index >= len(data)):
		raise ValueError('extractData() error: Index must be > 0 and < length of data.')

	targetpos = data[index][0]		# Position of generated targets for user
	path = data[index][1]			# Path of mouse during test

	x,y,t = [],[],[]				# Space and time of mouse path
	for i in range(0,len(path)):
		x.append(path[i][0])
		y.append(path[i][1])
		t.append(path[i][2])

	return([targetpos, [x,y,t]])


# Gets derivatives of path and smooths and plots if desired
def getMetrics(xlist,ylist,tlist,nsmooth, plotdata = False):
	# X,Y velocity,acceleration and magnitudes
	velx, vely, accx, accy, velmag, accmag = [], [], [], [], [], []

	# Calculate velocity
	for i in range(0,len(xlist)-1):
		dx = xlist[i+1] - xlist[i]
		dy = ylist[i+1] - ylist[i]
		dt = tlist[i+1] - tlist[i] 

		velx.append(dx/dt)
		vely.append(dy/dt)
		velmag.append(np.sqrt((velx[-1])**2 + (vely[-1])**2))

	# Smooth velocity 
	vxsmooth = savgol_filter(velx, nsmooth, 3)# smoothing window = nsmooth, spline order = 3
	vysmooth = savgol_filter(vely, nsmooth, 3)
	velmagsmooth = savgol_filter(velmag, nsmooth, 3)

	# Calculate acceleration
	for i in range(0,len(xlist)-2):
		accx.append((velx[i+1] - velx[i])/(tlist[i+1] - tlist[i]))
		accy.append((vely[i+1] - vely[i])/(tlist[i+1] - tlist[i]))
		accmag.append((velmag[i+1] - velmag[i])/(tlist[i+1] - tlist[i]))

	# Smooth acceleration
	axsmooth = savgol_filter(accx, nsmooth, 3)
	aysmooth = savgol_filter(accy, nsmooth, 3)
	accmagsmooth = savgol_filter(accmag, nsmooth, 3)

	if (plotdata):
		plt.figure()
		plt.plot(xlist, ylist, 'k.')
		plt.plot(xlist[0], ylist[0], 'go')
		plt.plot(xlist[len(xlist) - 1], ylist[len(ylist) - 1], 'ro')
		plt.grid(True)
		plt.xlabel('X (px)')
		plt.ylabel('Y (px)')
		plt.title('Mouse position')

		plt.figure()
		plt.plot(tlist[1:], velmagsmooth)
		plt.xlabel('Time (s)')
		plt.ylabel('Velocity Magnitude (px/s)')
		plt.title('Mouse velocity with ' + str(nsmooth) + ' point smoothing ')
		plt.grid(True)

		plt.figure()
		plt.plot(tlist[2:], accmagsmooth)
		plt.xlabel('Time (s)')
		plt.ylabel('Acceleration Magnitude (px/s/s)')
		plt.title('Mouse acceleration with ' + str(nsmooth) + ' point smoothing ')
		plt.grid(True)

		plt.show()

	return([vxsmooth, vysmooth, axsmooth, aysmooth])

# Finds closest point of x,y path to a target x,y point
def getClosestPoint(xlist, ylist, xtarget, ytarget):
	ds_min = 9E9
	for i in range(0,len(xlist)):
		dx = xlist[i] - xtarget
		dy = ylist[i] - ytarget
		ds = np.sqrt(dx**2 + dy**2)
		if (ds < ds_min):
			ds_min = ds
			ind_min = i
	return ind_min


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
		c2 = np.exp(-1.0/(x2/rampinwidth))
	else:
		c2 = 0.0

	return(c1*c2)

# Plots a spline given the nodes of the spline and the coefficients of the parametric representation of the curves (xc, yc) and # of points to plot with
# xc, yc represents x(t) = xc[0] + xc[1]*t + xc[2]*t^2 + ... for t c[0.0, 1.0]
# Oscillations adds sinusoidal oscillations to the path normal dicated by frequency list (radians) and amplitude list (pixels)
# Ramp adds exponential damping to oscillations near beginning and end of path
def plotSpline(nodelist, xc, yc, npoints, show = False, oscillations = False, freqlist = [], amplist = [], ramp = False, rampin = 0, rampout = 0, rampinwidth = 0, rampoutwidth = 0):
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
			for i in range (1,len(xc)):
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

		x.append(xsum)
		y.append(ysum)

	plt.figure()
	plt.plot(x,y, 'r-')
	
	# Plot spline node locations
	# Make all nodes blue except start node (green) and end node (red)
	colors     = ['bo']*len(nodelist)
	colors[0]  = 'go'
	colors[-1] = 'ro'

	for i in range(0,len(nodelist)):
		node = nodelist[i]
		plt.plot(node.x, node.y, colors[i])

	if show:
		plt.show()
	return([x,y])

def getDeviations(xpath, ypath, xspline, yspline, startind, plot = False):
	newx = xpath[startind:]
	newy = ypath[startind:]

	alldevx = []
	alldevy = []

	for i in range(0,len(newx)):
		x = newx[i]
		y = newy[i]
		cind = getClosestPoint(xspline, yspline, x, y)

		devx = [xspline[cind], x]
		devy = [yspline[cind], y]
		alldevx.append(devx)
		alldevy.append(devy)
		if plot:
			plt.plot(devx,devy, 'b-')

	return([alldevx, alldevy])
	

# Plot mouse path from test run # (index) as well as spline approximation of path from middle to end given previous path information (derivatives) 
# Spline is genereated from point closest to blue target until the last point in the path with degree set by the continuity at start node (1st deriv. continuity gives quadratic spline)
def plotPaths(data, index, continuity, oscillations = False, freqlist = [], amplist = [], ramp = False, rampin = 0, rampout = 0, rampinwidth = 0, rampoutwidth = 0):
	[[targetx, targety], [x,y,t]] = extractData(data,index)

	[vx, vy, ax, ay] = getMetrics(x,y,t,smoothwindow, False)

	# Location of middle blue target
	xblue = targetx[1]
	yblue = targety[1]

	# Start spline at the point closest to blue target
	ind_start = getClosestPoint(x,y, xblue, yblue)
	ind_end = len(x) - 1

	startnode = gs.Node()
	startnode.setPos(x[ind_start], y[ind_start])

	# Converting from space/time derivatives to space/parameter derivatives. dt_dtau = (t_f - t_i)/(tau_f - tau_i) = (t_f - t_i)/1.0
	dt_dtau = t[ind_end] - t[ind_start]

	# Set continuity at start node. Nth order derivative must be scaled by dt_dtau**n 
	if   (continuity == 1):
		startnode.setDeriv(1, [vx[ind_start]*dt_dtau, vy[ind_start]*dt_dtau])
	elif (continuity == 2):
		startnode.setDeriv(1, [vx[ind_start]*dt_dtau, vy[ind_start]*dt_dtau])
		startnode.setDeriv(2, [ax[ind_start]*dt_dtau**2, ay[ind_start]*dt_dtau**2])
	elif (continuity != 0):
		raise ValueError ('Start node spline derivative continuity must be between 0 and 2.')


	endnode = gs.Node()
	endnode.setPos(x[ind_end], y[ind_end])

	# Get spline polynomial coefficients for x,y
	[spline_xcoeffs, spline_ycoeffs] = gs.makeSpline([startnode, endnode])

	[xspline, yspline] = plotSpline([startnode, endnode], spline_xcoeffs, spline_ycoeffs, 300, oscillations = oscillations, freqlist = freqlist, amplist = amplist,
		ramp = ramp, rampin = rampin, rampout = rampout, rampinwidth = rampinwidth, rampoutwidth = rampoutwidth)

	plt.plot(x, y, 'k.')
	getDeviations(x, y, xspline, yspline, ind_start, True)
	plt.legend(['Spline approximation', 'Start node', 'End node', 'Mouse path', 'Deviation'])
	plt.axis('scaled')
	plt.show()


# Open pickled mouse paths 
with open(datapath, 'rb') as file:
	data = pickle.load(file)

# Get test information (time, # of runs, version)
header = data[0]

for i in range(1, len(data)):
	# Generate parameters random sinusoidal variations
	freq = []
	amp = []

	for j in range(0,3):
		freq.append(random.random()*3.0 + 1.0)  # Random frequencies between 1 and 4
		amp.append(random.random()*10.0 + 5.0) # Random amplitudes between 5 and 15

	plotPaths(data, i, 2, oscillations = True, freqlist = freq, amplist = amp, ramp = True, rampin = 0.1, rampout = 1.0, rampinwidth = 0.4, rampoutwidth = 0.2)

