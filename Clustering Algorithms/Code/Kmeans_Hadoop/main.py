"""main.py"""
import os

def readInput(name):
    datalist = []
    centroids = []
    dataset = open(name, 'r')
    for line in dataset:
        line = line.strip()
        datalist.append(line)
    for data in datalist:
        centroids.append(datalist.split('\t'))
    return centroids

def mapReduceHadoop():
    #k value
    k = 3
    #oldCentroids
    oldCentroids = []
    #newCentroids
    newCentroids = []
    iterationNo = 1
    maxIteration = 100
    #set the initial file name here
    f_name = "randomCentroids.txt"
    
    if(iterationNo == 1):
        #Call the mapper reducer to run for the first itertion
        os.system("hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.6.4.jar - file mapper.py -mapper mapper.py -file reducer.py -reducer reducer.py -file old_centroids.txt -input "+f_name+"-output newcentroids"+str(iteration))
        iteration += 1
        os.system("$HADOOP_HOME/bin/hdfs dfs -get newcentroids"+str(iteration)+"output"+str(iteration))
        oldCentroids = readInput(newcentroids+str(iteration).txt)
    else:
        while(iteration <= maxIteration):
            
            if(oldCentroids != newCentroids):
                oldCentroids = readInput(newcentroids+str(iteration).txt)
                os.system("hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-2.6.4.jar - file mapper.py -mapper mapper.py -file reducer.py -reducer reducer.py -file old_centroids.txt -input "+f_name+"-output newcentroids"+str(iteration))
                iteration += 1
                os.system("$HADOOP_HOME/bin/hdfs dfs -get newcentroids"+str(iteration)+"output"+str(iteration))
                newCentroids = readInput(newcentroids+str(iteration).txt)
        
