import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats as stat


############################################################################
######### Library for Conducting SFA on a 2d Matrix ########################
############################################################################

# Matrix should be in form np.array([[x(t)],[y(t)]...[z(t)]])

# Returns Matrix with 0 mean and Variance
def norm(arr):
	m = np.mean(arr,1)
	arr = arr-m[:,None]
	arr = arr/np.sqrt(np.mean(np.square(arr)))
	return arr

# Returns same matrix
def linExpand(arr):
	return arr

# Returns quadratic expansion of Matrix
# Ex. matrix [x1,x2,x3] -> matrix [x1,x2,x3,x1x1,x1x2,x2x2,x1x3]
def quadExpand(arr):
	s = arr.shape
	l = int(s[0] + s[0]*(s[0]+1)/2)
	out = np.zeros((l,s[1]))
	count = 0
	for i in range(s[0]):
		out[i] = arr[i]
		count = count + 1
	for i in range(s[0]):
		for j in range(i+1):
			out[count] = np.multiply(arr[i],arr[j])
			count = count + 1
	return out

# Conducts PCA Whitening on a Matrix to return matrix
# with 0 mean and identity covariance
# Also Removes Redundant Eigenvectors (eig < 10^-9)
def PCA(arr):
	m = np.mean(arr,1)
	arr = arr-m[:,None]
	cov = np.cov(arr)
	cov = cov + np.identity(cov.shape[0]) *0.0000001
	U, S, Vh = np.linalg.svd(cov)
	V = Vh.transpose
	where = np.argwhere(S > 0.00000011)
	siz = where.size
	S = 1/np.sqrt(S)
	S = S[:siz]
	S = S * np.identity(S.size)
	xrot = np.matmul(U.T[0:siz,:],arr)
	xwhite = np.matmul(S,xrot)
	return xwhite

# Returns weight vectors for SFA input-output functions

def weights(arr, retain, mode = 'retain'):
	dx = np.diff(arr, axis = 1)
	dcov = np.cov(dx)
	dcov = dcov + np.identity(dcov.shape[0]) *0.0000001
	U, S, Vh = np.linalg.svd(dcov)
	V = Vh.transpose
	S = S[S.size-retain:S.size]
	S = 1/np.sqrt(S)
	S = S * np.identity(S.size)
	if (mode ==	 'retain'):
		U = U[:,U.shape[1]-retain:U.shape[1]]
	return U

def SFAquad_WithoutTools(arr,j):
	xnorm = norm(arr)
	xquad = quadExpand(xnorm)
	xpca = PCA(xquad)
	xweights = weights(xpca, j)
	return np.matmul(xweights.T, xpca)


############################################################################
############## VISUALIZATION TOOLS USEFUL FOR SFA ##########################
############################################################################

# Simply plots a 1d array against its indices within the set bounds
def plot1d(arr,start,end,mode = 'line'):
	arr = arr[start:end]
	if mode == 'line':
		plt.plot(arr)
		plt.show()
	elif mode == 'scatter':
		plt.scatter(np.linspace(0,arr.size,arr.size),arr)
		plt.show()

# Plots a 2d array composed of 1d time dependent arrays within time bounds
def plot2d(arr,start,end):
	arr = arr[:,start:end]
	plt.imshow(arr)
	plt.show()

# Plots 2 arrays against each other and returns the r-value/p-value of a
# linear regression of the resultant graph
# Modes usable for the graph are 'line','scatter',and'none'
def correlateOne(arr1,arr2,start,end, mode = 'line'):
	arr1 = arr1[start:end]
	arr2 = arr2[start:end]
	if mode == 'line':
		plt.plot(arr1,arr2)
		plt.show()
	elif mode == 'scatter':
		plt.scatter(arr1,arr2)
		plt.show()
	slope, intercept, rval, pval, stderr = stat.linregress(arr1,arr2)
	return np.array([rval,pval])

# Graphs an array against a reference set of arrays (stored as a 2d array
# Returns the r-values and p-values for the linear regressions on the
# resultant graphs.
def correlate(mat,arr,start,end):
	mat = mat[:,start:end]
	arr = arr[start:end]
	leng = mat.shape[0]
	out = np.zeros((leng,2))
	for i in range(leng):
		slope, intercept, rval, pval, stderr = stat.linregress(mat[i],arr)
		out[i,0] = rval
		out[i,1] = pval
	return out

# Simple correlation between 1d arrays within a time bound without the
# usage of any phase shifts
def xcorr(arr1,arr2,start,end):
	arr1 = arr1[start:end]
	arr2 = arr2[start:end]
	return np.correlate(arr1,arr2)


