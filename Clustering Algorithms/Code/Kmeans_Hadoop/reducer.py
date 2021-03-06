#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys

current_key = None
old_list = None
key = None


#process the values sent from the mapper(cluster_id,gene_id,attributes)
for line in sys.stdin:
    line = line.strip()
    inputLine = line.split('\t')
    inputAttributes = []
    length = 0
    bline = []
    for i in range(0,len(inputLine)):
        if i == 0:
            key = inputLine[i]
        elif i == 1:
            gene = inputLine[i]
        else:
            aline = inputLine[i].replace("[","").replace("]","")
            aline = aline.split(",")
            length = len(aline)
            for a in aline:
                b = float(a.strip())
                bline.append(b)
                
    if not current_list:
        current_list = []
        for q in range(0,length):
            current_list[q] = []
            current_list[q].append(bline[q])
            
    if current_key == key:
        for k in range(0,length):
            current_list[k].append(bline[k])
    else:
        if current_key:
            pass
        current_key = key
        tempList = []
        for q in range(0,length):
            temp_list[q] = []
            temp_list[q].append(bline[q])
        current_list = temp_list  
            
#for each cluster_id compute the new  centroid value and output it
if current_key == key:
    mean = []
    index = 0
    sumCount = 0
    while(index < len(current_list[0])):
        for i in range(0,len(current_list)):
            sumCount += current_list[i][index]
            index += 1
        mean.append(sumCount/len(current_list))
    print(current_key+"\t"+str(mean))