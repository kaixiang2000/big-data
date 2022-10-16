from __future__ import print_function
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
# checking if the trip duration is more than a minute, trip distance is more than 1 miles, 
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
    #Task 1 Simple Linear Regression
    task1_sum_x=taxilinesCorrected.map(lambda x:x[5]).reduce(lambda x,y: x+y)
    task1_sum_y=taxilinesCorrected.map(lambda x:x[11]).reduce(lambda x,y: x+y)
    task1_sum_xy=taxilinesCorrected.map(lambda x:x[11]*x[5]).reduce(lambda x,y:x+y)
    task1_sum_xx=taxilinesCorrected.map(lambda x:x[5]*x[5]).reduce(lambda x,y: x+y)
    n=float(taxilinesCorrected.count())
    m_hat=(n*task1_sum_xy-(task1_sum_x*task1_sum_y))/(n*task1_sum_xx-(task1_sum_x*task1_sum_x))
    b_hat=(task1_sum_xx*task1_sum_y-(task1_sum_x*task1_sum_xy))/(n*task1_sum_xx-(task1_sum_x*task1_sum_x))
    result1=sc.parallelize([m_hat,b_hat])
    result1.coalesce(1).saveAsTextFile(sys.argv[2])
    sc.stop()
