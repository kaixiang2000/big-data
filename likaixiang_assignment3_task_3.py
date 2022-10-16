from __future__ import print_function
import findspark
findspark.init()
import numpy as np
import os
import sys
import requests
from operator import add

from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *
spark=SparkSession.builder.getOrCreate()
sc=SparkContext.getOrCreate()
sqlContext=SQLContext(sc)

#Exception Handling and removing wrong datalines
def isfloat(value):
    try:
        float(value)
        return True
 
    except:
         return False

#Function - Cleaning
#For example, remove lines if they don’t have 16 values and 
# checking if the trip distance and fare amount is a float number
# checking if the trip duration is more than a minute, trip distance is more than 1 mile, 
# fare amount and total amount are more than 1 dollars
def correctRows(p):
        if (len(p) == 17):
            if (isfloat(p[5]) and isfloat(p[11]) and isfloat(p[4]) and isfloat(p[15])):
                if (float(p[4])>= 2*60 and float(p[4])<= 60*60
                    and float(p[11]) >= 3 and float(p[11]) <= 200 
                    and float(p[5]) >= 1 and float(p[5]) <= 50
                    and float(p[15]) >= 3):
                    return p
#Main
if __name__ == "__main__":
#     if len(sys.argv) != 4:
#         print("Usage: main_task1 <file> <output> ", file=sys.stderr)
#         exit(-1)
    
    testDataFrame = spark.read.format('csv').options(header='false', inferSchema='true', sep =",").\
    load(sys.argv[1])
    testRDD = testDataFrame.rdd.map(tuple)
    taxilinesCorrected = testRDD.filter(correctRows)    
    #Task3 Fit Multiple Linear Regression using Gradient Descent
    task3=taxilinesCorrected.map(lambda x: 
                                    (float(x[16]),np.array([float(x[4])/60, float(x[5]), float(x[11]), float(x[12]),1])))
    beta=np.ones(5)*0
    learningRate = 0.000001
    num_iteration = 50
    n = float(task3.count())
    cost_old=0
    task3.cache()
    for i in range(num_iteration):
        
        gradientCost=task3.map(lambda x: (x[1], (x[0] - np.dot(x[1],beta)))).map(lambda x: (x[0]*x[1], x[1]**2 )).reduce(lambda x, y: (x[0] +y[0], x[1]+y[1]))
        
        cost= gradientCost[1]
        
        gradient=(-1/n)* gradientCost[0]
        if (i >= 1):
            if(cost < cost_old):
                learningRate = learningRate * 1.05
            
            else:
                learningRate = learningRate * 0.5
            
        cost_old = cost
        print("iteration=",i," Cost= ", cost,"learningrate =",learningRate)
        print("m=",beta[:4],"b=",beta[-1])
        beta = beta - learningRate * gradient