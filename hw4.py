"""
hw4.py
Name(s):
NetID(s):
Date:
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def longTime(A):
	w,v = np.linalg.eig(A)
	domEig = max(w)
	maxine = np.where(w == domEig)
	domVec = v[:,maxine]
	return(domEig, domVec)

def getAk(A,k):
	w,v = np.linalg.eig(A)

	P = v
	Pinv = np.linalg.inv(P)

	Dk = [[0.0 for i in range(len(A))] for j in range(len(A))]
	for i in range(len(Dk)):
		Dk[i][i] = w[i]**k

	AK = np.matmul((np.matmul(P,Dk)),Pinv)
	return AK

def error(uCurr, uLong):
	diff = uCurr - uLong
	err = np.linalg.norm(diff)
	return err

def normalize(v):
	v = v/np.linalg.norm(v)
	return v

def simulate(A,k,u0):
	sim = np.zeros((len(u0),k))
	Akn = np.identity(len(A))
	sim[:,0, None] = u0

	for i in range(1,k):
		Akn = np.matmul(Akn, A)
		u = np.matmul(Akn, u0)
		sim[:,i,None] = u
	return sim

"""
main function
"""


def main():

	A = np.array([[2,1],[1,1]])
	# print(longTime(A))
	# print(getAk(A,2))

	# v1 = np.array([[0.5**(-1/2)],[0.5**(-1/2)]])
	v2 = np.array([[1],[0]])
	# print(error(v1,v2))

	# v = np.array([[1],[1]])
	# print(normalize(v))

	print(simulate(A,3,v2))

	# Scenario A: Frogs
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
	print("u250 = ")
	print(u250)
	print("us = ")
	print(us)

	er = np.linalg.norm(u250-us) / np.linalg.norm(u250)
	print(er)

	#Plotting

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

	u250norm = normalize(u250)
	err = np.zeros((250, 1))
	for i in range(250):
		usnorm = normalize(sim_mat[:,i,None])
		err[i,0,None] = error(usnorm, u250norm)

	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(x, err, label = "Error vs. Time")
	plt.title("Error vs. Time")
	plt.xlabel("Time (iterations)")
	plt.ylabel("Error")
	plt.savefig("frogerror.png", bbox_inches = "tight")
	plt.close("all")

	# Scenario B: Owls
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
	print("u250 = ")
	print(u250)
	print("us = ")
	print(us)

	er = np.linalg.norm(u250-us) / np.linalg.norm(u250)
	print(er)

	#Plotting

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

	u250norm = normalize(u250)
	err = np.zeros((250, 1))
	for i in range(250):
		usnorm = normalize(sim_mat[:,i,None])
		err[i,0,None] = error(usnorm, u250norm)

	plt.figure()
	fig, ax = plt.subplots()
	ax.plot(x, err, label = "Error vs. Time")
	plt.title("Error vs. Time")
	plt.xlabel("Time (iterations)")
	plt.ylabel("Error")
	plt.savefig("owlerror.png", bbox_inches = "tight")
	plt.close("all")

if __name__ == '__main__':
	main()