import itertools as it

#read data from the file
def loadDataFromFile():
    datalist = []
    data = open('associationruletestdata.txt', 'r')
    for eachLine in data:
            datalist.append(eachLine)
    
    return datalist

#add gene number to the datset
def addGeneNumber(datalist):
    
    masterlist=[]     
    for data in datalist:
        new=[]
        item = data.split('\t')
        count = 1
        for i in range(0,len(item)-1):
            gene = item[i]
            gene = 'G'+str(count)+'_'+gene
            count += 1
            new.insert(i,gene)
        
        new.append(item[len(item)-1].split('\n')[0])
        masterlist.append(new)
    return masterlist

#initially create a list of all possible candidates-C1
def createCandidates(masterlist):
   
    allcandidates = [frozenset([candidate]) for sublist in masterlist for candidate in sublist]
                 
    return list(set(allcandidates))

#create a dictionary with candidate as key and list of transaction ids as values
def genTransaction(mainset,dataset):
    resdict = {}
    for ms in mainset:
        alist = []
        for i in range(0,len(dataset)):
            datarow = set(dataset[i])
            if ms.issubset(datarow):
                alist.append(i)
        resdict[ms] = alist
   
    return resdict

#calculate support
def calculatesupport(count,total,minsupport):
    
    support = count/total
    if support >= minsupport:
        return True,support
    else:
        return False,support

#generate L1 by performing set intersection of the passed candidates,check support and add the itemset if its frequent
def genFrequentItemSet(candidates,dataset,candidaterows,total,minsupport):
    frequentitemlist=[]
    supportresult = {}
    for candidate in candidates:
        finalset = set()
        count = 0
        if len(candidate) != 1:
            candidate = set(candidate)
            for item in candidate:
                aset = set(candidaterows.get(frozenset([item])))
                if (count == 0):
                    finalset = aset
                    count += 1
                if finalset == None:
                    break
                else:
                    finalset = finalset & aset
        
            test,supportcount =  calculatesupport(len(finalset),total,minsupport)
            if test:
                frequentitemlist.insert(0,frozenset(candidate))
            if supportcount != 0:
                supportresult[frozenset(candidate)] = supportcount
        else:
            supp = len(candidaterows.get(candidate))
            test,supportcount =  calculatesupport(supp,total,minsupport)
            if test:
                frequentitemlist.insert(0,frozenset(candidate))
            supportresult[frozenset(candidate)] = supportcount   
    return frequentitemlist,supportresult


#extending the frequent item set from length k to k+1 for every iteration by set union
def gencombinations(clist, lenk):
    setcombinations = []
    lastIndex = lenk - 2
    
    for i in range(len(clist)):
        for j in range(i+1,len(clist)): 
            
            firstList = list(clist[i])
            firstList = firstList[:lastIndex]

            firstList.sort()
        
            secondList = list(clist[j])
            secondList = secondList[:lastIndex]
            secondList.sort()
            
            if firstList == secondList: 
                setcombinations.append(clist[i]|clist[j])
    return setcombinations

# input data and run apriori frequent itemset generation
def apriorifrequentitemsetgen():
    datalist = loadDataFromFile()
    dataset = addGeneNumber(datalist)
    minsup = int(input("Minimum Support in terms of percentage: "))
    while(minsup < 0 and minsup > 100):
        minsup = float(input("Minimum Support in terms of percentage: "))
    minsup = minsup/100
    minconf = int(input("Minimum Confidence in terms of percentage: "))
    while(minconf < 0 and minconf > 100):
        minconf = float(input("Minimum Support in terms of percentage: "))
    minconf = minconf/100
    frequentItemSetList = []
    supportMatrix = {}
    primarycandidates = createCandidates(dataset)
    #print(len(primarycandidates))
    transactionlists = genTransaction(primarycandidates,dataset)
    #print(len(transactionlists))
    total = len(dataset)
    #generate support of frequent itemset of length 1
    itemlist,supportData = genFrequentItemSet(primarycandidates,dataset,transactionlists,total,minsup)
    #length of itemset generated
    itemsetlength = 1
    frequentItemSetList.append(itemlist)
    supportMatrix.update(supportData)
    #length of itemset to be generated
    itemsetlength += 1
    while(len(itemlist)>0):
        possiblecandidates = gencombinations(itemlist,itemsetlength)
        itemlist,supp = genFrequentItemSet(possiblecandidates,dataset,transactionlists,total,minsup)
        supportMatrix.update(supp)
        frequentItemSetList.append(itemlist)
        itemsetlength += 1
    return frequentItemSetList,supportMatrix,minconf

frequentItemSetList,supportMatrix,confidence = apriorifrequentitemsetgen()
frequentItemSets = frequentItemSetList[1:]    #rule gen requires itemsets of length 2 and higher
allRules = []
allCount = 0
for a in range(0,len(frequentItemSetList)):
    print("Length of "+str(a+1)+" frequent candidate itemset is: "+ str(len(frequentItemSetList[a])))
    allCount += len(frequentItemSetList[a])
print("Length of all item frequent candidate itemset is "+ str(allCount))

#generate subsets for given frequent itemset
def generateSubsets(rulesetitems):
    subsets = []
    for i in range(1, len(rulesetitems)):
        for combo in it.combinations(rulesetitems, i):
            subsets.append(list(combo))
    return list(map(set,subsets))

#calculate confidence
def calculateconfidence(num,den):
    
    confvalue = supportMatrix[num]/supportMatrix[den]
    if confvalue >= confidence:
        return True,confvalue
    else:
        return False,confvalue

# if confidence passes the test then add and generate the rule for the given pair
def generateRules(allsubsets,totalsets):
        for subset in allsubsets:
            num = totalsets
            den = totalsets-subset
            check,conf = calculateconfidence(num,den)
            if check:
                rhs = frozenset(subset)
                rule = (den,rhs, conf)
                print (den,'=======>',rhs,'confidence:',conf)
                allRules.append(rule)

#for every frquent itemset generate association rule
def aprioricandidaterulgen():
    for frequentItems in frequentItemSets:
        for frequentitem in frequentItems:
            ruleItems = set(frequentitem)
            subsetitems = generateSubsets(ruleItems)
            generateRules(subsetitems,frequentitem)

aprioricandidaterulgen()
rules = allRules

#template1
def template1(arg1,arg2,arg3):
    m=0
    for k in arg3:
        arg3[m]= k.split('_')[0] +'_'+k.split('_')[1].capitalize()
        m+=1

    count = 0
    template1 = []
    
    print('---------------------------------------')
    print('TEMPLATE1')
    print('---------------------------------------')
    
    if(arg1.upper()=='RULE'):
        
        if(type(arg2)==int):
            for i in rules:
                flag=0
                for j in arg3:
                    if(j in list(i[0]) or j in list(i[1])):
                        flag +=1
                
                if(flag == int(arg2)):
                    print(str(list(i[0]))+' --> '+ str(list(i[1])))
                    template1.append(i)
                    count+=1
        
        elif(arg2.upper()=='ANY'):
            for i in rules:
                flag = 0
                for j in arg3:
                    if (j in list(i[0]) or j in list(i[1])):
                        flag = 1
                        break
                        
                if(flag==1):
                    print(str(list(i[0]))+' --> '+ str(list(i[1])))
                    template1.append(i)
                    count+=1       
                        
            
        elif(arg2.upper()=='NONE'):
            for i in rules:
                flag=0
                for j in arg3:
                    if(j not in list(i[0]) and j not in list(i[1])):
                        flag +=1
                
                if(flag == len(arg3)):
                    print(str(list(i[0]))+' --> '+ str(list(i[1])))
                    template1.append(i)
                    count+=1
            
        else: print('Use only NUMBER/ANY/NONE in RULE')
    
    elif(arg1.upper()=='HEAD'):
        
        if(type(arg2)==int):
            for i in rules:
                flag=0
                for j in arg3:
                    if(j in list(i[0])):
                        flag +=1
                
                if(flag == int(arg2)):
                    print(str(list(i[0]))+' --> '+ str(list(i[1])))
                    template1.append(i)
                    count+=1
        
        elif(arg2.upper()=='ANY'):
            for i in rules:
                flag = 0
                for j in arg3:
                    if (j in list(i[0])):
                        flag = 1
                        break
                        
                if(flag==1):
                    print(str(list(i[0]))+' --> '+ str(list(i[1])))
                    template1.append(i)
                    count+=1     
                    
            
        elif(arg2.upper()=='NONE'):
            for i in rules:
                flag=0
                for j in arg3:
                    if(j not in list(i[0])):
                        flag +=1
                
                if(flag == len(arg3)):
                    print(str(list(i[0]))+' --> '+ str(list(i[1])))
                    template1.append(i)
                    count+=1
            
        
        
        else: print('Use only RULE/ANY/NONE in HEAD')
        
    elif(arg1.upper()=='BODY'):
    
        if(type(arg2)==int):
            for i in rules:
                flag=0
                for j in arg3:
                    if(j in list(i[1])):
                        flag +=1
                
                if(flag == int(arg2)):
                    print(str(list(i[0]))+' --> '+ str(list(i[1])))
                    template1.append(i)
                    count+=1
        
        elif(arg2.upper()=='ANY'):
            for i in rules:
                flag = 0
                for j in arg3:
                    if (j in list(i[1])):
                        flag = 1
                        break
                        
                if(flag==1):
                    print(str(list(i[0]))+' --> '+ str(list(i[1])))
                    template1.append(i)
                    count+=1     
            
        elif(arg2.upper()=='NONE'):
            for i in rules:
                flag=0
                for j in arg3:
                    if(j not in list(i[1])):
                        flag +=1
                
                if(flag == len(arg3)):
                    print(str(list(i[0]))+' --> '+ str(list(i[1])))
                    template1.append(i)
                    count+=1
            
        
        else: print('Use only RULE/ANY/NONE in BODY')
        
    else: print('Invalid Query')
    
    return template1,count
        
        
#template2
def template2(arg1,arg2):

    #m=0
    #for k in arg3:
        #arg3[m]= k.split('_')[0] +'_'+k.split('_')[1].capitalize()
        #m+=1   
    
    count = 0
    template2 = []
    print('---------------------------------------')
    print('TEMPLATE2')
    print('---------------------------------------')
    
    if(arg1.upper()=='RULE'):
        for i in rules:
            if(len(i[0])+len(i[1]) >= arg2):
                print(str(list(i[0]))+' --> '+str(list(i[1])))
                template2.append(i)
                count+=1
                
    elif(arg1.upper()=='HEAD'):
        for i in rules:
            if(len(i[0]) >= arg2):
                print(str(list(i[0]))+' --> '+ str(list(i[1])))
                template2.append(i)
                count+=1
                
    elif(arg1.upper()=='BODY'):
        for i in rules:
            if(len(i[1]) >= arg2):
                print(str(list(i[0]))+' --> '+ str(list(i[1])))
                template2.append(i)
                count+=1
                
    else:
        print('Invalid Query')
    
    return template2,count
    
#template3
def template3(arg1, *args):
    
    m=0
    for k in arg3:
        arg3[m]= k.split('_')[0] +'_'+k.split('_')[1].capitalize()
        m+=1

    template3 = []
    count=0
    
    
    if(arg1 == '1or1'):
        result1,count1 = template1(args[0],args[1],args[2])
        result2,count2 = template1(args[3],args[4],args[5])
        result3 = union(result1,result2)
        print('---------------------------------------')
        for i in result3:
            print(str(list(i[0]))+' --> '+ str(list(i[1])))
            template3.append(i)
            count+=1
    
    elif(arg1 == '1and1'):
        result1,count1 = template1(args[0],args[1],args[2])
        result2,count2 = template1(args[3],args[4],args[5])
        result3 = intersection(result1,result2)
        print('---------------------------------------')
        for i in result3:
            print(str(list(i[0]))+' --> '+ str(list(i[1])))
            template3.append(i)
            count+=1
        
    elif(arg1 == '1or2'):
        result1,count1 = template1(args[0],args[1],args[2])
        result2,count2 = template2(args[3],args[4])
        result3 = union(result1,result2)
        print('---------------------------------------')
        for i in result3:
            print(str(list(i[0]))+' --> '+ str(list(i[1])))
            template3.append(i)
            count+=1
        
    elif(arg1 == '1and2'):
        result1,count1 = template1(args[0],args[1],args[2])
        result2,count2 = template2(args[3],args[4])
        result3 = intersection(result1,result2)
        print('---------------------------------------')
        for i in result3:
            print(str(list(i[0]))+' --> '+ str(list(i[1])))
            template3.append(i)
            count+=1
        
    elif(arg1 == '2or2'):
        result1,count1 = template2(args[0],args[1])
        result2,count2 = template2(args[2],args[3])
        result3 = union(result1,result2)
        print('---------------------------------------')
        for i in result3:
            print(str(list(i[0]))+' --> '+ str(list(i[1])))
            template3.append(i)
            count+=1
        
    elif(arg1 == '2and2'):
        result1,count1 = template2(args[0],args[1])
        result2,count2 = template2(args[2],args[3])
        result3 = intersection(result1,result2)
        print('---------------------------------------')
        for i in result3:
            print(str(list(i[0]))+' --> '+ str(list(i[1])))
            template3.append(i)
            count+=1
        
    else: print('Invalid argument 1')
        
    return template3,count
             
        
def intersection(list1, list2): 
    list3 = [value for value in list1 if value in list2] 
    return list3 


def union(list1, list2): 
    list3 = list1 + list2 
    return list(set(list3))