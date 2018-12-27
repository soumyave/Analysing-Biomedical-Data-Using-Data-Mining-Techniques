import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def loadData(dataSet):
    masterList = []
    points = []
    groundTruth = []
    for eachLine in dataSet:
        data = []
        data = eachLine.split('\t')
        data[-1] = data[-1].split('\n')[0]
        #print(data[-1])
        masterList.append(data)   
    dataPoints = np.array(masterList, dtype='float')
    #print(masterList)
    rows,cols = dataPoints.shape
    for i in range(0,rows):
        points.append(dataPoints[i:i+1, 2:cols][0])
        groundTruth.append(int(dataPoints[i:i+1, 1:2][0]))
    #print(points)
    #print(groundTruth)
    return points,groundTruth

def DBSCAN(data,eps,minPts):
    C=0
    status = ['unvisited'] * rows        
    for P in range(0,rows):
        if(status[P]=='unvisited'): 
            status[P]=='visited'
            neighborPts=regionQuery(P, eps)
            if(len(neighborPts)<minPts):
                status[P]=-1
            else:
                C+=1
                expandCluster(P,neighborPts,C,eps,minPts,status)
    #print(status)
    return status

def regionQuery(P,eps):
    neighbors=set()
    neighbors.add(P)
    for i in range(0,rows):
        if (np.linalg.norm(data[P] - data[i])<eps):
            neighbors.add(i)
    return list(neighbors)

def expandCluster(P,neighborPts,C,eps,minPts,status):
    status[P]=C
    i=0
    while i<len(neighborPts):
        Q=neighborPts[i]
        if(status[Q]=='unvisited'):
            status[Q]=C
            neighborhood= regionQuery(Q,eps)
            #print(neighborhood)
            if(len(neighborhood)<minPts):
                i+=1
                continue
            if(len(neighborhood)>=minPts):
                neighborPts=list(neighborPts+neighborhood)
        i=i+1

def get_matrix(values,data):
    cluster_matrix = np.zeros((len(data),len(data)))
    for i in range(0,len(data)):
        for j in range(0,len(data)):
            if values[i]==values[j]:
                cluster_matrix[i][j]=1
                
    return cluster_matrix

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
    plt.title("DBSCAN Clustering")
    plt.legend(legend_val,unique_category)
    plt.xlabel("Component_1")
    plt.ylabel("Component_2")
    plt.show()


filename = input('Enter the filename : ')
dataSet = open(filename, 'r')
data,groundTruth = loadData(dataSet)
eps = float(input('Enter the eps value : '))
minPts = int(input('Enter the minPts value : '))
rows = len(data)
clusterList = DBSCAN(data,eps,minPts)
jaccard, rand = external_index(groundTruth,clusterList,data)
pca = PCA(n_components = 2)
transformed_data = pca.fit_transform(data)
scatter_plot(transformed_data,clusterList)
print('\tJACCARD COEFFICIENT : {} \n\tRAND INDEX : {}'.format(jaccard,rand))