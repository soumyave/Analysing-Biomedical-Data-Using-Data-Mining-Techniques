#K means clustering
import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def readInput():
    datalist = []
    filename = input('Enter the file to read : ')
    dataset = open(filename, 'r')
    for line in dataset:
        datalist.append(line)
    return datalist

def processInput(datalist):
    inputData = {}
    originalList = {}
    for data in datalist:
        gene = data.split('\t')
        attributes = []
        if originalList.get(gene[1]) == None:
            originalList[gene[1]] = [gene[0]]
        else:
            l = originalList.get(gene[1])
            l.append(gene[0])
            originalList[gene[1]] = l
        for i in range(2,len(gene)):
            if i == len(gene)-1:
                gene[i] = gene[i].split('\n')[0]
            attributes.append(gene[i])
        inputData[gene[0]] = attributes 
        attributecount = len(gene)-2
    return inputData,originalList,attributecount

#Euclidean distance
def Euclideandistance(first,second):

    a = np.array(first,dtype=float)
    b = np.array(second,dtype=float)
    dist = np.linalg.norm(a-b)
    return dist

def Euclideandistance2(first,second):

    a = tuple(first)
    b = tuple(second)
    dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return dist

def calculateCluster(inputd,centres,k,it):
   
    clusters = []
    for a in range(0,k):
        clusters.append([])
        
    if it == 1:
        for i in range(0,k):
            clusters[i].append(centres[i])
        
    for attributes in inputd:
    # add random centres to the respective clusters
        distance = []
        if(attributes not in [str(x) for x in centres]):
            for centre in centres:
                firstpoint = inputd.get(attributes)
                if it == 1:
                    secondpoint = inputd.get(str(centre))
                else:
                    secondpoint = centre
                distance.append(Euclideandistance(firstpoint,secondpoint))
            mindist = min(distance)
            i = distance.index(mindist)
            clusters[i].append(attributes)    
    return clusters 

#calculate new centres
def calculateCentroid(data,clusterPoints,attributecount):
    mean = []
    clustersize = len(clusterPoints)
    #print(clustersize)
    length = attributecount
    #print(length)
    for i in range(0,length):
        total = 0
        for clusterPoint in clusterPoints:
            attribute = data.get(clusterPoint)
            total += float(attribute[i])
        mean.append(round(total/clustersize,4))   
    
    return mean

def kmeans():
    global dataSet
    inputList = readInput()
    processedData,originalClusters,attributelength = processInput(inputList)
    dataSet = inputList
    kvalue = int(input("Value of k is: "))
    maxvalue = len(processedData)
    centerList = []
    clusterList = []
    for i in range(0,kvalue):
        randomCentre = randint(1,maxvalue)
        if randomCentre not in centerList:
            centerList.append(str(randomCentre))
    iteration = 1
    clusterList = calculateCluster(processedData,centerList,kvalue,iteration)
    oldClusters = clusterList
    newClusters = []
    count = 1
    while(oldClusters != newClusters):
        count += 1
        centerList = []
        for i in range(0,kvalue):
            centerList.append(calculateCentroid(processedData,clusterList[i],attributelength))
        oldClusters = clusterList
        clusterList = calculateCluster(processedData,centerList,kvalue,0)
        newClusters = clusterList
    results = outputList(oldClusters,maxvalue)
    #print(results) 
    return results
    
        
def outputList(clusteredResults,maxval):
    index = 0
    resultList = []
    for i in range(0,maxval):
        resultList.append(0)
    for i in range(0,len(clusteredResults)):
        l = i + 1
        for gene in clusteredResults[i]:
            resultList[int(gene)-1] = l
    return resultList   
        

def loadData(dataSet):
    masterList = []
    points = []
    groundTruth = []
    for eachLine in dataSet:
        data = []
        data = eachLine.split('\t')
        data[-1] = data[-1].split('\n')[0]
        masterList.append(data)   
    dataPoints = np.array(masterList, dtype='float')
    rows,cols = dataPoints.shape
    for i in range(0,rows):
        points.append(dataPoints[i:i+1, 2:cols][0])
        groundTruth.append(int(dataPoints[i:i+1, 1:2][0]))
    return points,groundTruth

def external_index(groundTruth,status,data):
    same_cluster_calculated = get_matrix(status,data)
    same_cluster_ground_truth = get_matrix(groundTruth,data)
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
    rand_index = math.ceil(m00+m11)/(m00+m11+m01+m10)
    return round(jaccard_coefficient,3),round(rand_index,3)

def get_matrix(values,data):
    cluster_matrix = np.zeros((len(data),len(data)))
    for i in range(0,len(data)):
        for j in range(0,len(data)):
            if values[i]==values[j]:
                cluster_matrix[i][j]=1
                
    return cluster_matrix

def scatter_plot(transformed_data,status):
    x = transformed_data[:,0] 
    y = transformed_data[:,1]
    plt.figure(figsize=[12,8]) 
    unique_category = np.unique(status).astype(int)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_category)))
    legend_val = []
    for uc in range(len(unique_category)):
        for val in range(len(transformed_data)):
            if status[val] == unique_category[uc]:
                x_current = x[val]
                y_current = y[val]
                scatter_plot = plt.scatter(x_current, y_current, c=colors[uc], s=70)
        legend_val.append(scatter_plot)
    plt.title("K-Means Clustering")
    plt.legend(legend_val,unique_category)
    plt.xlabel("Component_1")
    plt.ylabel("Component_2")
    plt.show()

def visualize(clusterList):
    data,groundTruth = loadData(dataSet)
    jaccard, rand = external_index(groundTruth,clusterList,data)
    pca = PCA(n_components = 2)
    transformed_data = pca.fit_transform(data)
    scatter_plot(transformed_data,clusterList)
    print('\tJACCARD COEFFICIENT : {} \n\tRAND INDEX : {}'.format(jaccard,rand))


results = kmeans()
visualize(results)