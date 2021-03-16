"""
hw4.py
Name(s): Ethan Donecoff, Arvind Parthasarathy
NetID(s): edd24, ap427
Date:03/16/2021
"""

import numpy as np
import matplotlib.pyplot as plt
import math

"""
longTime

This function returns the dominant eigenvalue and eigenvector for a given
matrix. 

INPUTS:
A: a 2d square numpy array

OUTPUTS:
(domEig, domVec): a 2-tuple consisting of domEig, the dominant eigenvalue, and 
domVec, an nx1 array representing the corresponding eigenvector
"""
def longTime(A):
	w,v = np.linalg.eig(A)
	domEig = max(w)
	maxind = np.where(w == domEig)
	maxind = maxind[0] # must do this to return a normal column vector below
	domVec = v[:,maxind] # column of v corresponding to max eigenvalue
	return(domEig, domVec)

"""
getAk

This function returns the value of A^k, where A is a matrix and k is a number.
The function first diagonalizes the matrix A into P, D, and Pinv where D is
a diagonal matrix of eigenvalues. The function then computes D^k by raising the
eigenvalues to the power of k, and finally returns AK = P * D * Pinv.

INPUTS:
A: a 2d square numpy array
k: a number representing the exponent

OUTPUTS:
AK: a numpy array representing A^k
"""
def getAk(A,k):
	w,v = np.linalg.eig(A)

	P = v
	Pinv = np.linalg.inv(P)

	Dk = [[0.0 for i in range(len(A))] for j in range(len(A))] # preallocate Dk
	for i in range(len(Dk)):
		Dk[i][i] = w[i]**k # raise diagonal elements to power k

	AK = np.matmul((np.matmul(P,Dk)),Pinv) #multiplication to get AK
	return AK

"""
error

This function takes the normalized difference between two unit vectors.

INPUTS:
uCurr: a unit vector (nx1 or 1xn numpy array)
uLong: a unit vector (nx1 or 1xn numpy array)

OUTPUTS:
err: a number representing the error between uCurr and uLong
"""
def error(uCurr, uLong):
	diff = uCurr - uLong
	err = np.linalg.norm(diff)
	return err

"""
normalize

This function normalizes a vector.

INPUTS:
v: a 1xn or nx1 numpy array representing a vector.

OUTPUTS:
v: a 1xn or nx1 numpy array with magnitude 1 and the same direction as the
input vector.
"""
def normalize(v):
	v = v/np.linalg.norm(v)
	return v

"""
simulate

This function computes u = (A^k)*u0 and stores u for each iteration of k
as a column vector in the matrix sim.

INPUTS:
A: a 2d square numpy array
k: a number representing the exponent
u0: a column vector (nx1 numpy array) representing the initial condition

OUTPUTS:
sim: an nxk numpy array where n is the length of u0. Each column c represents
(A^c)*u0, where c ranges from 0 to k.
"""
def simulate(A,k,u0):
	sim = np.zeros((len(u0),k)) # preallocate sim matrix as zeroes
	Akn = np.identity(len(A)) # preallocate Akn as identity
	sim[:,0, None] = u0 # first column of sim is u0

	for i in range(1,k): # compute product for each iteration
		Akn = np.matmul(Akn, A)
		u = np.matmul(Akn, u0)
		sim[:,i,None] = u # store u as column vector in sim matrix
	return sim

"""
main function

The main function contains some tests. It also runs calculations for two
scenarios: frogs and owls. 

For each simulation, we print the expected growth rate and 
stable population fraction. We compute the population at 
iteration 250 and compare it to a simulation using
a relative error. Then, we plot the stage-class model populations
vs. time along with the error between the simulation and the stable
population fraction vs. time.
"""
def main():

	# Testing
	"""
	A = np.array([[2,1],[1,1]])
	print(longTime(A))
	print(getAk(A,4))

	v1 = np.array([[0.5**(-1/2)],[0.5**(-1/2)]])
	v2 = np.array([[1],[0]])
	print(error(v1,v2))

	v = np.array([[1],[1]])
	print(normalize(v))

	print(simulate(A,4,v2))
	"""

	# Scenario A: Frogs
	print("\n Scenario A: Frogs \n")
	frogMat = [[0.0, 0.0, 3.0,  8.0], \
			   [0.4, 0.0, 0.0,  0.0], \
			   [0.0, 0.5, 0.0,  0.0], \
			   [0.0, 0.0, 0.25, 0.0]]
	frogInit = [[0.0],[0.0],[0.0],[250.0]]

	rate, pop = longTime(frogMat)
	print("The expected growth rate is")
	print(rate)
	print("\n The expected stable population fraction is")
	print(pop)

	Ak = getAk(frogMat, 250)
	u250 = np.matmul(Ak, frogInit)

	sim_mat = simulate(frogMat, 250, frogInit)
	us = sim_mat[:,249, None]
	print("\n u250 = ")
	print(u250)
	print("\n us = ")
	print(us)

	re = np.linalg.norm(u250-us) / np.linalg.norm(u250)
	print("\n The relative error is ")
	print(re)

	# Frog Plotting

	x = [i for i in range(250)]
	x = np.array(x)
	x.transpose()

	eggs = sim_mat[0,:,None]
	tadpoles = sim_mat[1,:,None]
	metamorphs = sim_mat[2,:,None]
	frogs = sim_mat[3,:,None]

	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(x, eggs, label = "Eggs vs. Time")
	ax.plot(x, tadpoles, label = "Tadpoles vs. Time")
	ax.plot(x, metamorphs, label = "Metamorphs vs. Time")
	ax.plot(x, frogs, label = "Adult Frogs vs. Time")
	legend = ax.legend(loc = 'upper left')
	plt.title("Frog Population Stages vs. Time")
	plt.xlabel("Time (iterations)")
	plt.ylabel("Population")
	plt.savefig("frogsimulation.png", bbox_inches = "tight")
	plt.close('all')

	# Frog Error
	err = np.zeros((250, 1))
	for i in range(250):
		usnorm = normalize(sim_mat[:,i,None])
		err[i,0,None] = error(usnorm, pop)

	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(x, err, label = "Error vs. Time")
	plt.title("Frogs: Error vs. Time")
	plt.xlabel("Time (iterations)")
	plt.ylabel("Error")
	plt.savefig("frogerror.png", bbox_inches = "tight")
	plt.close("all")

	# Scenario B: Owls
	print("\n Scenario B: Owls \n")
	owlMat = [[0.2,  0.1,  0.4,  1/3], \
			  [0.4,  0.4,  0.2,  1/3], \
			  [0.2,  0.3,  0.2,  1/3], \
			  [0.01, 0.01, 0.01, 1.5]]
	owlInit = [[100.0],[100.0],[0.0],[0.0]]

	rate, pop = longTime(owlMat)
	print("The expected growth rate is")
	print(rate)
	print("\n The expected stable population fraction is")
	print(pop)

	Ak = getAk(owlMat, 250)
	u250 = np.matmul(Ak, owlInit)

	sim_mat = simulate(owlMat, 250, owlInit)
	us = sim_mat[:,249, None]
	print("\n u250 = ")
	print(u250)
	print("\n us = ")
	print(us)

	re = np.linalg.norm(u250-us) / np.linalg.norm(u250)
	print("\nThe relative error is ")
	print(re)

	# Owls Plotting

	x = [i for i in range(250)]
	x = np.array(x)
	x.transpose()

	loc1 = sim_mat[0,:,None]
	loc2 = sim_mat[1,:,None]
	loc3 = sim_mat[2,:,None]
	loc4 = sim_mat[3,:,None]

	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(x, loc1, label = "Location 1 vs. Time")
	ax.plot(x, loc2, label = "Location 2 vs. Time")
	ax.plot(x, loc3, label = "Location 3 vs. Time")
	ax.plot(x, loc4, label = "Location 4 vs. Time")
	legend = ax.legend(loc = 'upper left')
	plt.title("Owl Location Populations vs. Time")
	plt.xlabel("Time (iterations)")
	plt.ylabel("Population at Location")
	plt.savefig("owlsimulation.png", bbox_inches = "tight")
	plt.close('all')

	#Owls Error

	err = np.zeros((250, 1))
	for i in range(250):
		usnorm = normalize(sim_mat[:,i,None])
		err[i,0,None] = error(usnorm, pop)

	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(x, err, label = "Error vs. Time")
	plt.title("Owls: Error vs. Time")
	plt.xlabel("Time (iterations)")
	plt.ylabel("Error")
	plt.savefig("owlerror.png", bbox_inches = "tight")
	plt.close("all")

if __name__ == '__main__':
	main()