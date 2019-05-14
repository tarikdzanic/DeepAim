import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter
import math
import generatesplines as gs
import random

smoothwindow = 51

def extractData(data, index):
	if (index < 1 or index >= len(data)):
		raise ValueError('extractData() error: Index must be > 0 and < length of data.')

	targetpos = data[index][0]
	path = data[index][1]

	x,y,t = [],[],[]
	for i in range(0,len(path)):
		x.append(path[i][0])
		y.append(path[i][1])
		t.append(path[i][2])

	return([targetpos, [x,y,t]])


def getMetrics(xlist,ylist,tlist,nsmooth, plotdata = False):

	velx, vely, accx, accy, velmag, accmag = [], [], [], [], [], []

	for i in range(0,len(xlist)-1):
		dx = xlist[i+1] - xlist[i]
		dy = ylist[i+1] - ylist[i]
		dt = tlist[i+1] - tlist[i] 

		velx.append(dx/dt)
		vely.append(dy/dt)
		velmag.append(np.sqrt((velx[-1])**2 + (vely[-1])**2))

	vxsmooth = savgol_filter(velx, nsmooth, 3)# smoothing window = nsmooth, spline order = 3
	vysmooth = savgol_filter(vely, nsmooth, 3)
	velmagsmooth = savgol_filter(velmag, nsmooth, 3)

	for i in range(0,len(xlist)-2):
		accx.append((velx[i+1] - velx[i])/(tlist[i+1] - tlist[i]))
		accy.append((vely[i+1] - vely[i])/(tlist[i+1] - tlist[i]))
		accmag.append((velmag[i+1] - velmag[i])/(tlist[i+1] - tlist[i]))

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

def plotSpline(nodelist, xc, yc, npoints):
	t = np.linspace(0,1,npoints)
	x = []
	y = []
	for i in range(0,len(t)):
		curt = t[i]
		xsum = 0
		ysum = 0
		for j in range(0,len(xc)):
			xsum += xc[j]*curt**j
			ysum += yc[j]*curt**j
		x.append(xsum)
		y.append(ysum)

	plt.figure()
	plt.plot(t,x, 'k-')

	plt.figure()
	plt.plot(t,y, 'k-')

	plt.figure()
	plt.plot(x,y, 'k-')
	for i in range(0,len(nodelist)):
		node = nodelist[i]
		plt.plot(node.x, node.y, 'ro')
	#plt.show()
	

datapath = 'MouseMovementData_SessID05893'

with open(datapath, 'rb') as file:
	data = pickle.load(file)

header = data[0]

i = 2
nnodes = 2


[[targetx, targety], [x,y,t]] = extractData(data,i)
[vx, vy, ax, ay] = getMetrics(x,y,t,smoothwindow, False)

xblue = targetx[1]
yblue = targety[1]

ind_start = getClosestPoint(x,y, xblue, yblue)
ind_end = len(x) - 1

nodelist = []
node = gs.Node()
node.setPos(x[ind_start], y[ind_start])

dt = t[ind_end] - t[ind_start]
node.setDeriv(1, [vx[ind_start]*dt, vy[ind_start]*dt])
node.setDeriv(2, [ax[ind_start]*dt*dt, ay[ind_start]*dt*dt])

nodelist.append(node)
nodeindices = []

for i in range(0,nnodes-2):
	nodeindices.append(random.randint(ind_start+1, ind_end-1))

nodeindices = sorted(nodeindices)

for i in range(0, len(nodeindices)):
	nodeind = nodeindices[i]
	node = gs.Node()
	node.setPos(x[nodeind], y[nodeind])
	nodelist.append(node)

node = gs.Node()
node.setPos(x[ind_end], y[ind_end])
nodelist.append(node)

[xspline, yspline] = gs.makeSpline(nodelist)

plotSpline(nodelist, xspline, yspline, 300)
plt.plot(x, y, 'k.')
plt.show()
