import numpy as np
from tree import Node
from collections import Counter
from random import randint
classList = None
nominalIndexes = None
attributeCount = None
dataRecordCount = None
maxTreeDepth = None
minDataSplitSize = None
k_fold = None
fileName = None

#read line by line
#Input : File 
#Output : List of values line by line
def readInput(inputFileName):
    datalist = []
    dataset = open(inputFileName,'r')
    for line in dataset:
        datalist.append(line)
    return datalist

#input (Eg:List of values)
#output (List: list of  list of feature attributes including class it belongs to)
def formatInput(datalist):
    processedData = []
    for features in datalist:
        feature = features.split('\t')
        feature[-1] = feature[-1].split('\n')[0]#(class is stored)
        processedData.append(feature)  
    return processedData

#input : value
#output : boolean of converting the input value to float value
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

#input : List of list of attributes
#output : List of processed Attributes (mapped to continous and categorical values),index of nominal attributes
def processAttributes(datalist):
    nomAttr = set()
    for i in range(0,len(datalist[0])):
        res = isfloat(datalist[0][i])
        if not res:
            nomAttr.add(i)
            tempDict = {}
            counter = 0
            for data in datalist:
                if data[i] not in tempDict:
                    tempDict[data[i]] = counter
                    counter = counter + 1
            for data in datalist:
                data[i] = tempDict.get(data[i])
        else:
            for data in datalist:
                data[i] = float(data[i])
                if data[i].is_integer():
                    data[i] = int(data[i])
                    
    return datalist,nomAttr

#input : DataSet
#output : Set of class of dataset
def processClasses(datalist):
    classSet = set()
    for data in datalist:
        classSet.add(data[-1]) 
    return list(classSet)

#input : dataset
#output : classScore (P(j/t).pow(2))
def calculateClassScore(values):
    classScore = 0.0
    if len(values) != 0:
        for classVal in classList:
            count = 0
            for classValue in values:
                if(classValue[-1] == classVal):
                    count += 1
            pjt = count / len(values)
            doublepjt = pjt ** 2
            classScore += doublepjt
    return classScore

#input : left and right dataset
#output : weightedGiniIndex
def calculateWeightedGiniIndex(valueList1,valueList2):
    giniIndex = 0.0
    totalSize = len(valueList1) + len(valueList2)
    score1 = calculateClassScore(valueList1) 
    score2 = calculateClassScore(valueList2)
    weight1 = len(valueList1)/totalSize
    weight2 = len(valueList2)/totalSize
    weightedScore1 = (1.0 - score1) * weight1
    weightedScore2 = (1.0 - score2) * weight2
    giniIndex = weightedScore1 + weightedScore2
    return giniIndex

#input : training Data
#output : node with giniindex,split attribute index and value,split datasets
def getBestSplit(trainingdatalist):
    minginiIndex = 1.1
    resultNode = None
    random_attrs = set()
    while len(random_attrs) != randomFeatures :
        random_num = randint(0,(attributeCount-2))
        if random_num not in random_attrs:
            random_attrs.add(random_num)
        
    for i in random_attrs: 
        for data in trainingdatalist:
            attribute = data[i]
            branch1 = []
            branch2 = []
            if i in nominalIndexes:
                for row in trainingdatalist:
                    if row[i] == attribute:
                        branch1.append(row)
                    else:
                        branch2.append(row)
            else:
                for row in trainingdatalist:
                    if row[i] < attribute:
                        branch1.append(row)
                    else:
                        branch2.append(row)
            giniIndex = calculateWeightedGiniIndex(branch1,branch2)
            if giniIndex <= minginiIndex:
                minginiIndex = giniIndex
                resultNode = Node(attribute,i,minginiIndex,branch1,branch2)         
    return resultNode

#input : childNode dataset
#output : childNode with set predicted class label by majority voting
def buildLeafNode(child):
    classlabel = [row[-1] for row in child]
    label = max(set(classlabel), key=classlabel.count)
    return Node(childNode = True,predictlabel = label)

#input : Node to build child nodes and current depth of the tree
#output : New node
def buildChildNodes(node,currDepth):
    leftSet = node.left
    rightSet = node.right
    
    if len(leftSet) == 0:
        node.left = buildLeafNode(rightSet)
        node.right = node.left
        return
    
    if len(rightSet) == 0:
        node.left = buildLeafNode(leftSet)
        node.right = node.left
        return
        
    if currDepth >= maxTreeDepth:
        node.left = buildLeafNode(leftSet)
        node.right = buildLeafNode(rightSet)
        return 
    
    if len(leftSet) > minDataSplitSize:
        node.left = getBestSplit(leftSet)
        currDepth += 1
        buildChildNodes(node.left,currDepth)
    else:
        node.left = buildLeafNode(leftSet)
        
    if len(rightSet) > minDataSplitSize:
        node.right = getBestSplit(rightSet)
        currDepth += 1
        buildChildNodes(node.right,currDepth)
    else:
        node.right = buildLeafNode(rightSet)

#input : TrainingData
#output : rootNode of the decision tree
def constructDecisionTree(trainingSet):
    rootNode = getBestSplit(trainingSet)
    currentDepth = 1
    buildChildNodes(rootNode,currentDepth)
    return rootNode

# Make a prediction with a decision tree
def predictClass(node, row):
    #check if it is child node
    if node.childNode == False:
        #check if the attribute is of the nominal type
        if node.attr_index in nominalIndexes:
            if row[node.attr_index] == node.attr:
                #recurse left
                return predictClass(node.left,row)
            else:
                #recurse right
                return predictClass(node.right,row)
        #attribute is integer/continuous
        else:
            if row[node.attr_index] <= node.attr:
                return predictClass(node.left,row)
            else:
                return predictClass(node.right,row)
    #return the predicted label of child node
    else:
        return node.predictlabel

def randomForestAlgorithm(train,test):
    
    predictedClassList = []
    roots = []
    for i in range(trees):
        baggingList = []
        for index in range(len(train)):
            baggingList.append(train[randint(0,len(train)-1)])
        root = constructDecisionTree(baggingList)
        roots.append(root)
        
    for testdata in test:
        predictedClassLabel = []
        for root in roots:
            predictedClassLabel.append(predictClass(root, testdata))
        predictedClassList.append(Counter(predictedClassLabel).most_common(1)[0][0])
        
    return predictedClassList

#Function to calculate the performance measure
#input : predicted and actual class values in list
#output : dictionary with accuracy,F1,precision,recall values
def PerformanceMeasure(predicted_class, actual_class):
    performanceMetrics = {}
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
    
    if(total_data != 0):
        accuracy = (tp+tn)/total_data
    if (tp + fp != 0):
        precision = tp/(tp+fp) 
    if(tp+fn != 0):
        recall = tp/(tp+fn)
    if(recall+precision != 0):
        F1_measure = (2*recall*precision) / (recall+precision)
        
    performanceMetrics['Accuracy'] = accuracy
    performanceMetrics['Precision'] = precision
    performanceMetrics['Recall'] = recall
    performanceMetrics['F1_Measure'] = F1_measure
     
    print("Accuracy    :", performanceMetrics.get('Accuracy'))
    print("Precision   :", performanceMetrics.get('Precision'))
    print("Recall      :", performanceMetrics.get('Recall'))
    print("F-1 Measure :", performanceMetrics.get('F1_Measure'))
    return performanceMetrics

#input : EntireDataSet
#output : Accuracy,Precision,Recall,F1-Measures in a dictionary for each k-fold and Average for k-fold runs
def k_fold_validation(data):
    global k_fold
    k_fold= 10
    k_rows = int(len(data)/k_fold)
    kPerformanceMetrics = {}
    perfMetrics = {}
    averagePerfMetrics = {}
    rootOfTree = None
    total_accuracy,total_precision,total_recall,total_F1 = 0,0,0,0

    for i in range(0,k_fold):
        #Partitions dataset into training and testing dataset based on number of rows in each fold and number of iterations
        print("-----------------------"+str(i+1)+"-------------------------")
        start = i * k_rows 
        end =((i+1) * k_rows)- 1
        copy = list(data)
        test = copy[start:end+1]
        del copy[start:end+1]
        train=copy
        
        
        predicted_class = randomForestAlgorithm(train,test)
        test_class = [x[-1] for x in test]
        perfMetrics = PerformanceMeasure(predicted_class,test_class)
        kPerformanceMetrics[str(i+1)] = perfMetrics
        
        total_accuracy += perfMetrics.get('Accuracy')
        total_precision += perfMetrics.get('Precision')
        total_recall += perfMetrics.get('Recall')
        total_F1 += perfMetrics.get('F1_Measure')
        
        
    averagePerfMetrics['Accuracy'] = total_accuracy/k_fold
    averagePerfMetrics['Precision'] = total_precision/k_fold
    averagePerfMetrics['Recall'] = total_recall/k_fold
    averagePerfMetrics['F1_Measure'] = total_F1/k_fold
        
    kPerformanceMetrics['Average'] = averagePerfMetrics
    
    return kPerformanceMetrics   

def printPerformanceStatistics(metrics):
    tempDict = {}
    
    tempDict = metrics.get('Average')
    print("-----------------------------------------------")
    print("\nPerformance Measures for file :", fileName)
    print("Number of Data records :",dataRecordCount)
    print("Number of different attributes in each record :",attributeCount)
    print("-----------------------------------------------")
    print("\nAverage measures")
    print("Accuracy    :", tempDict.get('Accuracy'))
    print("Precision   :", tempDict.get('Precision'))
    print("Recall      :", tempDict.get('Recall'))
    print("F-1 Measure :", tempDict.get('F1_Measure'))
    
    
def main():
    global fileName
    fileName = input("Enter the name of input file(with extension)")
    inputList = readInput(fileName)
    global maxTreeDepth
    maxTreeDepth = int(input("Enter the depth of the tree(stop split criteria)"))
    global minDataSplitSize
    minDataSplitSize = int(input("Enter the minimum size of dataset(stop split criteria)"))
    global trees
    trees = int(input("Enter the number of trees : "))
    formatList = formatInput(inputList)
    global nominalIndexes
    preprocessedList,nominalIndexes = processAttributes(formatList)
    global classList
    classList = processClasses(formatList)
    global attributeCount
    attributeCount = len(preprocessedList[0]) - 1
    global randomFeatures
    randomFeatures = int((attributeCount-1) ** 0.5)
    global dataRecordCount
    dataRecordCount = len(preprocessedList)
    performanceStatistics = k_fold_validation(preprocessedList)
    printPerformanceStatistics(performanceStatistics)
    

main()