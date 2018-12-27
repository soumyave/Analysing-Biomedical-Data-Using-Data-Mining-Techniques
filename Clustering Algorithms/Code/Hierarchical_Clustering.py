# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:04:06 2018

@author: Soumya V
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

# Funtion to read the file and differentiate features and diseases attributes from the given data file
def load_data(input_file):
    with open(input_file) as textFile:
        file_data = [line.strip().split("\t") for line in textFile]    
    final_data = np.array(file_data)  
    genes_data = np.array(final_data[:,2:final_data.shape[1]],dtype=float)    
    ground_truth = final_data[:,1]
    return genes_data,ground_truth

#Function to find the two data points separated by a minimum distance that is greater than zero 
def find_min_value_pair(distance_matrix):
    
    min_value = sys.maxsize
    matrix_size = len(distance_matrix)
    for row_value in range(0, matrix_size):
        for col_value in range(row_value+1, matrix_size):
            if distance_matrix[row_value][col_value] < min_value and not(distance_matrix[row_value][col_value] == 0):
                min_value = distance_matrix[row_value][col_value]
                row = row_value
                col = col_value
                
    return row,col
                    
#Function to perform Hierarchical Clustering                    
def hierarchial_clustering(cluster_data,cluster_number,distance_matrix):
    
    actual_cluster_length = len(cluster_data)
    
    while cluster_number != len(cluster_data):
        
        row,col = find_min_value_pair(distance_matrix)
        
        #Recomputing the distance matrix
        for i in range(0,len(cluster_data)):  
            distance_matrix[i][row]=min(distance_matrix[i][row],distance_matrix[i][col])
            distance_matrix[row][i]=min(distance_matrix[row][i],distance_matrix[col][i])
         
        #Merging the data that belong to the same cluster
        cluster_data[row].extend(cluster_data[col])
        
        #Deleting the entry for one of the merged points from distance matrix to avoid reconsidering the same value during the other iterations
        distance_matrix = np.delete(distance_matrix,col,0)    
        distance_matrix = np.delete(distance_matrix,col,1)
        
        #Removing the entry for one of the merged points from the cluster data
        cluster_data.pop(col)
        
    #An array is computed with indices as each data point and same value is assigned for all data points belonging to the same cluster
    final_clusters = np.zeros(actual_cluster_length)
    cluster_size = len(cluster_data)
    for cluster in range(0,cluster_size):
        for element in cluster_data[cluster]:
            final_clusters[element] = cluster+1;
                
    return final_clusters

#Function to perform PCA using PCA module from sklearn decomposition library        
def pca(genes_data):
    pca = PCA(n_components = 2)
    transformed_data = pca.fit_transform(genes_data)
    return transformed_data

#Function to form a matrix with value 1 for data belonging to the same cluster
def get_matrix(values,total_data):
    
    cluster_matrix = np.zeros((len(genes_data),len(genes_data)))
    
    for i in range(0,len(genes_data)):
        for j in range(0,len(genes_data)):
            if values[i]==values[j]:
                cluster_matrix[i][j]=1
                
    return cluster_matrix
          
#Function to calculate the external indices Jaccard Coefficient and Rand Index
def external_index(ground_truth,final_clusters,genes_data):
    
    same_cluster_calculated = get_matrix(final_clusters,genes_data)
    same_cluster_ground_truth = get_matrix(ground_truth,genes_data)
    
    m00 = 0
    m11 = 0
    m10 = 0
    m01 = 0
    
    for i in range(0,len(same_cluster_calculated)):
        for j in range(0,len(same_cluster_calculated)):
            if same_cluster_ground_truth[i][j]==same_cluster_calculated[i][j] and same_cluster_ground_truth[i][j] == 1:
                m11+=1
            elif same_cluster_ground_truth[i][j]==same_cluster_calculated[i][j] and same_cluster_ground_truth[i][j] == 0:
                m00+=1
            elif same_cluster_ground_truth[i][j]!=same_cluster_calculated[i][j] and same_cluster_ground_truth[i][j] == 1:
                m10+=1
            else:
                m01+=1
                
    jaccard_coefficient = m11/(m11+m01+m10)
    rand_index = (m00+m11)/(m00+m11+m01+m10)
    
    return jaccard_coefficient,rand_index


# Function to generate scatter plots for passed data parameters
def scatter_plot(transformed_data,final_clusters):
    x = transformed_data[:,0] #taking first data column as x axis
    y = transformed_data[:,1] #taking second data column as y axis
    plt.figure(figsize=[12,8]) 
    
    unique_category = np.unique(final_clusters).astype(int) #finding the unique categories from cluster
    
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_category)))
    legend_val = []
    
    #Loop through the new attribute data calculated using the three methods and plot them based on the category they belong to
    for uc in range(len(unique_category)):
        for val in range(len(transformed_data)):
            if final_clusters[val] == unique_category[uc]:
                x_current = x[val]
                y_current = y[val]
                scatter_plot = plt.scatter(x_current, y_current, c=colors[uc], s=70)
        legend_val.append(scatter_plot)
        
    plt.title("Hierarchical Agglomerative Clustering")
    plt.legend(legend_val,unique_category)
    plt.xlabel("Component_1")
    plt.ylabel("Component_2")
    plt.show()



# Reads the file input
file_name = input("Enter file name with extension :")
cluster_number = int(input("Enter the number of clusters :"))

genes_data, ground_truth = load_data(file_name);

#If the number of clusters requested is greater than the total length of given data, it will stop execution
if cluster_number > len(genes_data):
    print("\n--The number of clusters requested is more than the given data--")

#If the number of clusters requested is not greater than the total length of given data, clustering will be performed    
else:
    cluster_data=[]
    
    for i in range(0, len(genes_data)):
        temp = []
        temp.append(i)
        cluster_data.append(temp)
    
    distance_matrix = euclidean_distances(genes_data,genes_data)
    
    final_clusters = hierarchial_clustering(cluster_data,cluster_number,distance_matrix)
    
    transformed_data = pca(genes_data)
    
    print("\n\tHIERARCHICAL AGGLOMERATIVE CLUSTERING : ",file_name)
    scatter_plot(transformed_data, final_clusters)
    
    jaccard_coefficient, rand_index = external_index(ground_truth,final_clusters,genes_data)
    
    print("\nJACCARD COEFFICIENT : ",jaccard_coefficient)
    print("\nRAND INDEX : ",rand_index)
