class Node(object):
    def __init__(self, attribute= None,index=None,gini=None,left = None,right = None,childNode = False,predictlabel = None):
        self.attr = attribute
        self.attr_index = index
        self.gini = gini
        self.left = left
        self.right = right
        self.childNode = childNode
        self.predictlabel = predictlabel