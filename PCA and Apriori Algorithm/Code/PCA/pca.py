# -*- coding: utf-8 -*-
"""
Created on  Wed Sep  1 13:33:08 2018

@author: Soumya V
"""

import sys
import numpy as np
from numpy import linalg as LA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD 
import matplotlib.pyplot as plt


# Funtion to read the file and differentiate features and diseases attributes from the given data file
def load_data(input_file):
    with open(input_file) as textFile:
        file_data = [line.strip().split("\t") for line in textFile]    
    final_data = np.array(file_data) 
    features = np.array(final_data[:,0:final_data.shape[1]-1],dtype=float)  
    diseases = final_data[:,-1]    
    return features,diseases


# Function to perform PCA.
def pca(features_data, disease_data):
    
    length = len(features_data)
    mean_value = features_data.mean(axis=0) #calculating mean for original data
    adjusted_data = features_data - mean_value #adjusting the original data based on mean
    data_transpose = np.transpose(adjusted_data)
    covariance_matrix = np.dot(data_transpose,adjusted_data)/length #computing covariance matrix on adjusted data
    eigen_values, eigen_vectors = LA.eig(covariance_matrix) #finding the eigen values and eigen vectors from covariance matrix
    eigen_values_max = eigen_values.argsort()[::-1][:2] #sorting eigen_values in decreasing order and taking top 2
    eigen_vectors_max = eigen_vectors[:,eigen_values_max] #taking the two eigen vectors corresponding to the eigen values
    transformed_data = np.dot(features_data,eigen_vectors_max) #computing new attributes
    scatter_plot(transformed_data,disease_data,'pca')

	
# Function to perform SVD using TruncatedSVD from sklearn decomposition library
def svd(features_data, disease_data):
    transformed_svd = TruncatedSVD(n_components=2).fit_transform(features_data) 
    scatter_plot(transformed_svd, disease_data,'svd')


# Function to perform TSNE using TSNE from sklearn manifold library
def tsne(features_data, disease_data):
    transformed_tsne = TSNE(n_components=2,init = 'pca',learning_rate=100).fit_transform(features_data)
    scatter_plot(transformed_tsne, disease_data,'tsne')


# Function to generate scatter plots for passed data parameters
def scatter_plot(plot_data,disease_array,plot_type):
    x = plot_data[:,0] #taking first data column as x axis
    y = plot_data[:,1] #taking second data column as y axis
    plt.figure(figsize=[15,10]) 
    
    unique_category = np.unique(disease_array) #finding the unique categories of diseases from given array
    
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_category)))
    legend_val = []
    
    #Loop through the new attribute data calculated using the three methods and plot them based on the category they belong to
    for uc in range(len(unique_category)):
        for val in range(len(x)):
            if disease_array[val] == unique_category[uc]:
                x_current = x[val]
                y_current = y[val]
                scatter_plot = plt.scatter(x_current, y_current, c=colors[uc])
        legend_val.append(scatter_plot)
        
    plt.title(plot_type.upper() + " : " + file_name)
    plt.legend(legend_val,unique_category)
    plt.xlabel("Component_1")
    plt.ylabel("Component_2")
    plt.show()

# Reads the file input
file_name = input("Enter file name with extension :")

features, diseases = load_data(file_name);

print("\n\t\t\tPrincipal Component Analysis (PCA)")
pca(features, diseases)
print("\n\t\t\tSingular Value Decomposition (SVD)")
svd(features, diseases)
print("\n\t\tt-Distributed Stochastic Neighbor Embedding (TSNE)")
tsne(features, diseases)
