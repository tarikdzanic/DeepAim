import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


# Node object for B-spline. Allows specifying x,y position of knots and derivatives at the nodes
class Node(object):
	def __init__(self):
		self.x = 0
		self.y = 0
		# Derivatives are structured so that the 1st value (deriv[0]) corresponds to first derivative value. Value = None is no derivative is specificed
		self.derivs = []
		self.degree = 0

	def setPos(self, x,y):
		self.x = x
		self.y = y
		self.setDegree() 	# Update degree of node at beginning

	# Add dorder-th derivative with given value to the list. 
	def setDeriv(self, dorder, value):
		# Make new list and add old deriv list values to the new list (Todo: Make this better, would be slow for large number of derivatives)
		newderivs = [None]*dorder

		for i in range(0,len(self.derivs)):
			newderivs[i] = self.derivs[i]

		# Add new derivative to list
		newderivs[dorder-1] = value

		self.derivs = newderivs
		self.setDegree()

	# Calculate degree of node
	def setDegree(self):
		self.degree = 1

		for i in range(0,len(self.derivs)):
			if (self.derivs[i] != None):
				self.degree += 1


# Make row for solving Ax=b spline equation for derivatives at node points given Ax=b row for position and order of derivative (deriv)
def deriveRow(row, deriv):
	newrow = []
	t = row[1] 		# Knot value is the second term in the position row after the constant term [1, t, t**2, t**3, etc.]

	for i in range(0,len(row)):
		# If deriving more times than degree of polynomial, set to 0
		if (i < deriv):
			newrow.append(0)	

		# Else evaluate derived polynomial terms
		else:
			exp = i - deriv 	# New exponent is original exponent - # of derivations
			const = 1

			# Calculate new constant term as truncated factorial 
			for j in range(0,deriv):
				const *= i-j			

			# Add new term to row
			newrow.append(const*(t**exp))
	return newrow

# Set up Ax = b matrix/vector for solving spline path given node, order of polynomial, parametric knot position (tvalue), and if solving for x (bool_x = True) or y (False)
def getAxb(node, polyorder, tvalue, bool_x):
	A, b = [], []

	# First value of b vector is either x or y coordinate, depending on bool_x
	if (bool_x):
		value = node.x
	else:
		value = node.y

	# Create row for position (c0 + c1*t + c2*t**2 + ... = x)
	firstrow = []
	for i in range(0,polyorder):
		firstrow.append(tvalue**i)

	A.append(firstrow)
	b.append(value)

	# Create rows for derivatives (ex: c1 + 2*c2*t + ... = dy/dt)

	for j in range(0, len(node.derivs)):

		deriv_value = node.derivs[j]
		deriv_degree = j + 1 #Since derivs[0] corresponds to first derivative

		if (deriv_value != None):

			row = deriveRow(firstrow, deriv_degree)
			A.append(row)

			# The system is underconstrained for dy/dx = dy/dt * dt/dx so just set dx/dt = 1 which gives dy/dt = dy/dx
			if (bool_x):
				b.append(deriv_value[0])
			else:
				b.append(deriv_value[1])

	return [A,b]

# Create parametric knot positions (t) between 0 and 1 based on relative distances between nodes
def getKnots(nodelist):
	distances = []
	s = 0

	for i in range(0,len(nodelist)-1):
		dx = nodelist[i+1].x - nodelist[i].x
		dy = nodelist[i+1].y - nodelist[i].y
		s += np.sqrt(dx**2 + dy**2)
		distances.append(s)
	
	totaldist = s
	knots = [0]

	for i in range(0,len(distances)):
		knots.append(distances[i]/totaldist)

	return knots

# Make spline given list of nodes
def makeSpline(nodelist): 
	# Create evenly spaced parametric knot positions between 0 and 1 for all the nodes
	knots = getKnots(nodelist)

	# Calculate polynomial degree of B-spline
	order = 0
	for i in range(0,len(nodelist)):
		node = nodelist[i]
		order += node.degree
	
	# Global At=b for x and y coordinates 
	A_x, A_y, b_x, b_y = [], [], [], []

	# Compute local At=b at each node and append to global At=b
	for i in range(0,len(nodelist)):
		node = nodelist[i]
		t = knots[i]

		[A_x_local, b_x_local] =  getAxb(node, order, t, True)
		[A_y_local, b_y_local] =  getAxb(node, order, t, False)
		A_x += A_x_local
		b_x += b_x_local
		A_y += A_y_local
		b_y += b_y_local

	# Solve for B-spline polynomial coeffs
	xcoeffs = scipy.linalg.solve(np.array(A_x), np.array(b_x))
	ycoeffs = scipy.linalg.solve(np.array(A_y), np.array(b_y))

	return([xcoeffs, ycoeffs])

