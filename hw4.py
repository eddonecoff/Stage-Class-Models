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
	domVec = v[np.where(w == domEig)]
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
	sim = np.ones((len(u0),k))
	Akn = np.eye(len(A))
	for i in range(len(u0)):
		sim[i][0] = u0[i][0]
	u = u0

	for i in range(1,k):
		print(sim)
		Akn = np.matmul(Akn, A)
		u = np.matmul(Akn, u)
		for j in range(len(u)):
			sim[j][i] = u[j][0]
		print(sim)
	return sim

"""
main function
"""


def main():

	# A = np.array([[2,1],[1,1]])
	# # print(longTime(A))
	# # print(getAk(A,2))

	# # v1 = np.array([[0.5**(-1/2)],[0.5**(-1/2)]])
	# v2 = np.array([[1],[0]])
	# # print(error(v1,v2))

	# # v = np.array([[1],[1]])
	# # print(normalize(v))

	# print(simulate(A,2,v2))

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
	us = sim_mat[:,249]
	print("u250 = ")
	print(u250)
	print("us = ")
	print(us)
	print(len(sim_mat))
	print(len(sim_mat[0]))

	# Scenario B: Owls
	owlMat = [[0.2,  0.1,  0.4,  1/3], \
			  [0.4,  0.4,  0.2,  1/3], \
			  [0.2,  0.3,  0.2,  1/3], \
			  [0.01, 0.01, 0.01, 1.5]]
	owlInit = [[100.0],[100.0],[0.0],[0.0]]

if __name__ == '__main__':
	main()