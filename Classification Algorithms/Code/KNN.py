# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 12:38:27 2018

@author: Soumya V
"""

import numpy as np
from scipy.spatial import distance
import operator

#Funtion to read data from file and convert nominal and continuous values to float
def load_data(filename):
    
    mapper = {}
    with open(filename) as textFile:
        file_data = [line.strip().split("\t") for line in textFile]
        
    #converting nominal(given as string) and continuous value(given as real values) to float
    for i in range(len(file_data)):
        for j in range(len(file_data[i])-1):
            if file_data[i][j].isalpha():
                if str(j)+file_data[i][j] not in mapper:
                    mapper[str(j)+file_data[i][j]] = float(len(mapper))
                file_data[i][j] = mapper[str(j)+file_data[i][j]]    
            else:
                file_data[i][j] = float(file_data[i][j])
                
        file_data[i] = np.array(file_data[i], dtype=float)
        
    return np.array(file_data)

#Function to compute euclidean distance, find the K nearest neighbors and predict the class label
def KNN(normalized_data, test_data, mean, std_dev, k):
    distance_list=[]
    nearest_neighbors = []
    nearest_neighbor_labels=[]
    results={}
    
    #Normalising traininig data using z-score normalisation
    normalized_test_data=(test_data[0:len(test_data)-1]-mean)/std_dev
    
    #Calculating distance between each test sample and all training samples
    for i in range(len(normalized_data)):
        euclidean_distance = distance.euclidean(normalized_data[i][0:len(normalized_data[0])-1],normalized_test_data)
        distance_list.append((normalized_data[i],euclidean_distance))
    
    distance_list.sort(key=operator.itemgetter(1))
    
    #Storing the k nearest neighbours and their labels in two seperate lists
    for i in range(k):
        nearest_neighbors.append(distance_list[i][0])
        nearest_neighbor_labels.append(distance_list[i][0][-1])
     
    #deriving the labels for test instances based on nearest neighbor labels and voting
    for i in range(len(nearest_neighbor_labels)):
        if not nearest_neighbor_labels[i] in results:
            results[nearest_neighbor_labels[i]] = 1
        else:
            results[nearest_neighbor_labels[i]]=results[nearest_neighbor_labels[i]]+1
         
    predicted_label = max(results.items(), key=operator.itemgetter(1))[0]
        
    return predicted_label

#Function to implement KNN
def normalize_and_perform_Knn(train_data,test_data,k):
    
    predicted_labels=[]
    train_data_set=np.array([line[0:-1] for line in train_data])
    train_labels=np.array([line[-1] for line in train_data])
     
    mean=train_data_set.mean(axis=0)
    std_dev=train_data_set.std(axis=0)
    
    #Normalising traininig data using z-score normalisation
    for i in range(len(train_data)):
        normalised_data = []
        for j in range(len(train_data[0])-1):
            normalised_data.append((train_data[i][j]-mean[j])/std_dev[j])
        normalised_data.append(train_labels[i])
        train_data[i]=normalised_data
    
    for elt in range(len(test_data)):
        predicted_labels.append(KNN(train_data ,test_data[elt],mean,std_dev,k))
        
    return np.array(predicted_labels)

#Function to calculate the performance measure
def PerformanceMeasure(predicted_class, actual_class):

    tp,tn,fp,fn=0,0,0,0
    accuracy, precision, recall, F1_measure=0,0,0,0
    
    total_data=len(predicted_class)
   
    for i in range (0,total_data):  
        if predicted_class[i]==actual_class[i] and actual_class[i]==1:
            tp+=1
        elif predicted_class[i]==actual_class[i] and actual_class[i]==0:
            tn+=1
        elif predicted_class[i]!=actual_class[i] and actual_class[i]==0:
            fp+=1
        else:
            fn+=1
                
    accuracy = (tp+tn)/total_data
    if (tp+fp)!=0:
        precision = tp/(tp+fp) 
    recall = tp/(tp+fn)
    if (recall+precision)!=0:
        F1_measure = (2*recall*precision) / (recall+precision)
    
    return accuracy, precision, recall, F1_measure

#Function to partition data into K disjoint subsets and use them for validation using KNN Algorithm
def k_fold_validation(data,k_value):
    
    k_fold=10
    k_rows = int(len(data)/k_fold)

    total_accuracy,total_precision,total_recall,total_F1 = 0,0,0,0

    for i in range(0,k_fold):
        #Partitions dataset into training and testing dataset based on number of rows in each fold and number of iterations
        start = i * k_rows 
        end = ((i+1) * k_rows) - 1
        copy = list(data)
        test = copy[start:end+1]
        del copy[start:end]
        train=copy
        
        test_labels=[line[-1] for line in test]
        
        predicted_class = normalize_and_perform_Knn(train,test,k_value)
        accuracy, precision, recall, F1_measure= PerformanceMeasure(predicted_class, test_labels)

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_F1 += F1_measure

    return total_accuracy/k_fold,total_precision/k_fold,total_recall/k_fold,total_F1/k_fold

#Gets the users choice
option = input("Do you have seperate train and test datasets :\n1.Yes\n2.No\nChoice : ")

#To implement KNN when seprate input files for train and test dataset is given
if option == '1' or option.lower() == "yes":
    #Reads the file input 
    train_file = input("Enter train file name with extension :")
    train_data = load_data(train_file)
    test_file = input("Enter test file name with extension :")
    test_data = load_data(test_file)
    #Reads input of number of nearest neighbors to retrieve
    k_value = input("Enter the value of K (number of nearest neighbors) : ")
    predicted_class = normalize_and_perform_Knn(train_data,test_data,int(k_value))
    Accuracy, Precision, Recall, F1_Measure= PerformanceMeasure(predicted_class, test_data[:,-1].astype(int))
    print("\nPerformance Measures for files :" + train_file + " , " + test_file)

#To perform 10 fold cross validation with single input file
elif option == '2':
    #Reads the file input 
    file_name = input("Enter file name with extension :")
    #Reads input of number of nearest neighbors to retrieve
    k_value = input("Enter the value of K (number of nearest neighbors) : ")
    original_data=load_data(file_name)
    Accuracy, Precision, Recall , F1_Measure = k_fold_validation(original_data,int(k_value))
    print("\nPerformance Measures for file :", file_name)

else:
    print("Invalid Input!!")
    
print("Accuracy    :", Accuracy)
print("Precision   :", Precision)
print("Recall      :", Recall)
print("F-1 Measure :", F1_Measure)