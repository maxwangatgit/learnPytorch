'''
Author: your name
Date: 2020-10-24 11:00:15
LastEditTime: 2020-10-27 12:25:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \learnPytorch\dascan.py
'''
# coding:UTF-8
import numpy as np
import random
import math
import copy
import scipy.io as sio
import matplotlib.pyplot as plt
import time



def entropyBase(p, n):
    """
    calculate th e information gain of two classes
    """
    p = float(p)
    n = float(n)
    temp1 = p/(p+n)
    temp2 = n/(p+n)
    
    # log can't work for 0
    temp1forLog = temp1+0.00001
    temp2forLog = temp2+0.00001
    return  round((-temp1*math.log2(temp1forLog) - temp2*math.log2(temp2forLog)),2)





def claculateEntropy(branchingDict):
    sum = 0.0
    sumList = []
    entropyList = []
    entropy = 0.0
    for dataGroup in branchingDict["value"]:        
        sumList.append(float(np.sum(dataGroup)))
        entropyList.append(entropyBase(dataGroup[0],dataGroup[1]))
    sum = float(np.sum(sumList))
    print(sum)
    for index, groupSum in enumerate(sumList):
        entropy = entropy + (groupSum/sum) * entropyList[index]

    print(" %-20s %-15s %-15s %-15s"%("name","pi","ni","I(pi, ni)"))
    for ii, groupEntropy in enumerate(entropyList):
        print("%-20s %-15.2f %-15.2f %-15.2f"% (branchingDict["key"][ii],branchingDict["value"][ii][0],branchingDict["value"][ii][1],groupEntropy))
    return  round(entropy,2)


def getGain(mainDict, branchingDict):
    dataOrigin = mainDict["value"][0]
    originEntropy  = entropyBase(dataOrigin[0],dataOrigin[1])
    newEntropy     = claculateEntropy(branchingDict)
    informationGain= originEntropy - newEntropy 
    return round(informationGain,2)


def decideTree():
    "base on 2007 exam"

    mainDict = {
        "key":["age"],
        "value":[[172, 332]]
    }
    majorDict = {
        "key": ["arts", "appl_science","science"],
        "value":[[172, 96], [0, 148], [0, 84]]
    }
    statusDict = {
        "key": ["Graduate", "Undergraduate"],
        "value":[[172, 0], [0, 332]]
    }

    print("**********************Based on Major******************************************")
    print("the information gain is %.2f"%getGain(mainDict,majorDict))
    print("\n")
    print("**********************Based on Status******************************************")
    print("the information gain is %.2f"%getGain(mainDict,statusDict))

decideTree()