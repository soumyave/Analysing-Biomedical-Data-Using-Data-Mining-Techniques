# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:13:59 2018

@author: Soumya V
"""

import numpy as np
import scipy.stats as sc

#Funtion to read data from file and seperate nominal and continuous values
def load_data(input_file):
    nominal_data=set()
    continuous_data=set()
    
    with open(input_file,'r') as textFile:
        file_data = [line.strip().split("\t") for line in textFile]  
        
    rows = file_data[0]
    
    #seperating nominal(given as string) and continuous value(given as real values)
    for i in range(0,len(rows)-1):
        if rows[i].isalpha() :
            nominal_data.add(i)
        else:
            continuous_data.add(i)
    
    return np.array(file_data), nominal_data,continuous_data
  
#Function to implement Naive Bayes validation
def NaiveBayes(train_data, test_data, nominal_data, continuous_data):
    
    predicted_class=[]
    continuous_0=[]
    continuous_1=[]
    nominal_0 = []
    nominal_1 = []
    nominal_size = len(nominal_data)
    continuous_size = len(continuous_data)
    
    total_0=0
    total_1=0
    
    columns_train = len(train_data)
    size_test = len(test_data)
    
    #seperating training dataset instances based on the class value(0/1)
    for i in range(0, columns_train):
        continuous_temp=[]
        nominal_temp=[]
        
        if int(train_data[i][-1]) == 0 :
            total_0 +=1
            for num in nominal_data:
                nominal_temp.append(train_data[i][num])
            nominal_0.append(nominal_temp)
            for num in continuous_data:
                continuous_temp.append(float(train_data[i][num]))
            continuous_0.append(continuous_temp)
            
        elif int(train_data[i][-1]) == 1 : 
            total_1 +=1
            for num in nominal_data:
                nominal_temp.append(train_data[i][num])
            nominal_1.append(nominal_temp)
            for num in continuous_data:
                continuous_temp.append(float(train_data[i][num]))
            continuous_1.append(continuous_temp)
    
    nominal_0 = np.array(nominal_0)
    nominal_1 = np.array(nominal_1)
    continuous_0 = np.array(continuous_0)
    continuous_1 = np.array(continuous_1)
    
    #finding mean and standard deviation for continuous values to help in calculating posterior probability
    if continuous_size > 0:
        mean_0 = continuous_0.mean(axis=0)
        mean_1 = continuous_1.mean(axis=0)
        std_dev_0 = continuous_0.std(axis=0)
        std_dev_1 = continuous_1.std(axis=0)
        
    if nominal_size > 0:
            nominal_data0 = [list(x) for x in zip(*nominal_0)]
            nominal_data1 = [list(x) for x in zip(*nominal_1)]
            
    prior_probability_0 = total_0 / float(total_1+total_0)
    prior_probability_1 = total_1 / float(total_1+total_0)

    #Predicting the probability for the test set
    for i in range(0,size_test):
        
        test_continuous=[]
        test_nominal=[]
        
        probability_0 = 1.0
        probability_1 = 1.0
        posterior_probability_0 = 1.0
        posterior_probability_1 = 1.0
        
        test_row=test_data[i]
        test_row_length = len(test_row)-1
        
        for num in range (0,test_row_length): 
            if num in nominal_data:
                test_nominal.append(test_row[num])
            elif num in continuous_data:
                test_continuous.append(float(test_row[num]))
        
        nominal_len = len(test_nominal)
        continuous_len = len(test_continuous)
        
        #calculating posterior probability for nominal features by counting through the features
        if nominal_size > 0:
            for i in range(0,nominal_len):
                posterior_probability_0 *= nominal_data0[i].count(test_nominal[i])/total_0
                posterior_probability_1 *= nominal_data1[i].count(test_nominal[i])/total_1
                                
        #using probability density function to calculate posterior probability for continuous feature
        if continuous_size > 0:
            for i in range(0,continuous_len):
                posterior_probability_0 *= sc.norm(mean_0[i], std_dev_0[i]).pdf(test_continuous[i])
                posterior_probability_1 *= sc.norm(mean_1[i], std_dev_1[i]).pdf(test_continuous[i])
                
                
        probability_0 = posterior_probability_0 * prior_probability_0
        probability_1 = posterior_probability_1 * prior_probability_1
        total_prob = probability_0 + probability_1

        #Finding the label(0/1) with highest probability and adding it to the predicted_class
        if probability_0 > probability_1:
            predicted_class.append(0)
        elif probability_1 > probability_0:
            predicted_class.append(1)
            
    return predicted_class, probability_0/total_prob,probability_1/total_prob


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
    precision = tp/(tp+fp) 
    recall = tp/(tp+fn)
    if (recall+precision)!=0:
        F1_measure = (2*recall*precision) / (recall+precision)
      
    return accuracy, precision, recall, F1_measure
            
#Function to partition data into K disjoint subsets and use them for validation using Naive Bayes Algorithm
def k_fold_validation(data):
    
    k_fold=10 
    k_rows = int(len(data)/k_fold)

    total_accuracy,total_precision,total_recall,total_F1 = 0,0,0,0

    for i in range(0,k_fold):
        #Partitions dataset into training and testing dataset based on number of rows in each fold and number of iterations
        start = i * k_rows 
        end = ((i+1) * k_rows) - 1
        test = data[start:end+1]
        copy = list(data)
        del copy[start:end]
        train=copy
        
        test_data = np.array(test)
        
        predicted_class, Prob_0, Prob_1 = NaiveBayes(train,test,nominal,continuous)
        accuracy, precision, recall, F1_measure= PerformanceMeasure(predicted_class, test_data[:,-1].astype(int))

        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_F1 += F1_measure

    return total_accuracy/k_fold,total_precision/k_fold,total_recall/k_fold,total_F1/k_fold
    
#Reads the file input
file_name = input("Enter file name with extension :")
data, nominal, continuous = load_data(file_name)

#Gets the users choice
option = input("Do you wish to Perform :\n1. Query Validation\n2. 10 fold Validation\nChoice : ")

#To calculate and display the class probabilities if user gives a query input
if option == '1':
    test_data = []
    given_query = input("Enter Query seperated by comma : ")
    query = given_query.split(',')
    query.append('1')
    np.array(query) 
    test_data.append(query)
    predicted_class,Prob_0,Prob_1 = NaiveBayes(data,test_data,nominal,continuous)
    print("\nProbabilities for query : X = {" + given_query + "}")
    print("Probability Class 0 - P(H0|X) : ", Prob_0)
    print("Probability Class 1 - P(H1|X) : ", Prob_1)
    print("Predicted Class : Class",predicted_class);
    
#To perform 10 fold cross validation and display the performance measures
elif option == '2':
    Accuracy,Precision,Recall,F1_Measure = k_fold_validation(data)
    print("\nPerformance Measures for file :", file_name)
    print("Accuracy    :", Accuracy)
    print("Precision   :", Precision)
    print("Recall      :", Recall)
    print("F-1 Measure :", F1_Measure)
    
else:
    print("Invalid Input!!")
