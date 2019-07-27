import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf 
import pyfilterbank.gammatone as g
import scipy.ndimage.filters as filt
from sklearn import svm
from sklearn.linear_model import Perceptron
from tqdm import tqdm

import SFA_Tools.SFA_Sets as s

# Read and load vocalizations from a list of files
def get_data(file_list):
    vocalizations = []
    rate = 0
    for f in file_list:
        vocal, rate = sf.read(f)
        vocalizations.append(vocal)
    return vocalizations

# Compute the average power of a signal
def signal_power(sig):
    return np.sum(np.square(sig))/sig.size

# Scales a noise vector to have a specified signal-to-noise ratio with
# another waveform
def scale_noise(vocalizations, noise, ratio):
    data = np.zeros(1)
    for vocal in vocalizations:
        data = np.hstack((data,vocal))
        
    initial_ratio = signal_power(data)/signal_power(noise[:data.size])
    return noise * np.sqrt(initial_ratio/ratio)

# Applies a gammatone transform with a given filter bank to a waveform
def gamma_transform(data, gfb):
    analysed = gfb.analyze(data)
    
    transformed = np.zeros((len(gfb.centerfrequencies),data.size))
    for i in range(len(gfb.centerfrequencies)):
        (band,state) = analysed.__next__()
        transformed[i] = abs(band)
        
    return transformed

# Applies a gammatone transform with given filter bank to a list of 
# different waveforms
def gamma_transform_list(data, filterbank):
    transformed = []
    
    for d in tqdm(data):
        d_transformed = gamma_transform(d, filterbank)
        transformed.append(d_transformed)
        
    return transformed

# Plots gammatone transformed vocalizations effectively
def plot_input(inp, name):
    plt.figure(figsize=(12,3))
    plt.title(name)
    plt.imshow(inp, aspect = 'auto')
    plt.show()
    return

# Gamma Function
def gamma(n, a, b, m):
    arr = np.arange(1,n+1)
    return a*np.power(arr,m)*(np.exp(-b*arr))

# Creates a temporal filter
def temporalFilter():
    arr = gamma(400,1.5,0.04,2) - gamma(400,1,0.036,2) 
    arr = arr / np.var(arr)
    return arr

# Applies list of temporal filters to a transformed vocal
def temporal_transform(data,filters):
    transformed = None
    init = True
    for f in filters:
        filtered = filt.convolve(data,f[:,None].T)
        if(init):
            transformed = filtered
            init = False
        else:
            transformed = np.vstack((transformed,filtered))
            
    return transformed

# Applies list of temporal filters to a list of transformed vocals
def temporal_transform_list(data,filters):
    transformed = []
    
    for d in tqdm(data):
        d_transformed = temporal_transform(d, filters)
        transformed.append(d_transformed)
        
    return transformed

# Applies SFA To Data
def getSF(data,name, mode = 'quad', retain = 20, transform = False):
    (data_normalized,mean,variance) = s.norm(data)
    print(name, ': Normalization Complete...')
    
    if (mode == 'quad'):
        data_normalized = s.quadExpand(data_normalized)
    print(name, ': Nonlinear Expansion Complete...')
    
    (data_Sphered,data_SS) = s.PCA(data_normalized)
    print(name, ': Sphering Complete...')
    
    weights = s.weights(data_Sphered, retain)    
    weights = np.flip(weights.T,0)
    print(name, ': Weights Determined...')
    
    if(transform):
        transformed = weights @ data_Sphered
        return transformed, mean, variance, data_SS, weights
    else:
        return mean, variance, data_SS, weights

# Tests SFA on Data
def testSF(data, name, mean, variance, SS, weights, mode = 'quad'):
    data_normalized = s.normTest(data, mean, variance) 
    print(name, ': Normalization Complete...')
    
    if (mode == 'quad'):
        data_normalized = s.quadExpand(data_normalized)
    print(name, ': Nonlinear Expansion Complete...')
    
    data_Sphered = s.PCATest(data_normalized,SS)
    print(name, ': Sphering Complete...')
    
    return weights @ data_Sphered

# Get Labels For Data
def getlabels(data):
    labels = None
    initialized = False
    for i,d in enumerate(data):
        if(not(initialized)):
            labels = np.zeros(d[0].size)
            initialized = True
        else:
            nextlabel = np.ones(d[0].size) * i
            labels = np.hstack((labels,nextlabel))
    return labels

# SFA Plot Classifiers
def SFAClassifiedPlot(features,classifier, labels, n = 500, figure_size = (10,7)):
    x_min, x_max = features[0].min() - 1, features[0].max() + 1
    y_min, y_max = features[1].min() - 1, features[1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n), np.linspace(y_min, y_max, n))
    
    arr = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(arr)
    Z = Z.reshape(xx.shape)
    
    labelset = list(set(labels))
    pos = []
    for label in labelset:
        positions = [i for i,x in enumerate(labels) if x == label]
        pos.append(positions)
        
    plt.figure(figsize=figure_size)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    for i,p in enumerate(pos):
        plt.scatter(features[0][p][::10], features[1][p][::10], c = 'C' + str(int(labels[p[0]])), cmap=plt.cm.Paired)
        
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show() 
    return
