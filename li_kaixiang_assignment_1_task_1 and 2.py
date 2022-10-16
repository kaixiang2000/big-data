import os
import sys
import requests
from operator import add

from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)


# Exception Handling and removing wrong datalines
def isfloat(value):
    try:
        float(value)
        return True

    except:
        return False


# For example, remove lines if they don’t have 16 values and
# checking if the trip distance and fare amount is a float number
# checking if the trip duration is more than a minute, trip distance is more than 0.1 miles,
# fare amount and total amount are more than 0.1 dollars
def correctRows(p):
    if (len(p) == 17):
        if (isfloat(p[5]) and isfloat(p[11])):
            if (float(p[4]) > 60 and float(p[5]) > 0.10 and float(p[11]) > 0.10 and float(p[16]) > 0.10):
                return p


# Main
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: main_task1 <file> <output> ", file=sys.stderr)
        exit(-1)
    testDataFrame = sqlContext.read.format('csv').options(header='true', inferSchema='true', sep=","). \
        load(sys.argv[1])

    testRDD = testDataFrame.rdd.map(tuple)
    # calling isfloat and correctRows functions to cleaning up data
    taxiLinesCorrected = testRDD.filter(correctRows)
    # task1 Top-10 Active Taxis
    task1 = taxiLinesCorrected.map(lambda x: (x[0],x[1]))
    result1=task1.groupByKey().map(lambda x:(x[0],len(x[1]))).top(10, lambda x: x[1])
    answerFor1 = sc.parallelize(result1)
    answerFor1.coalesce(1).saveAsTextFile(sys.argv[2])
    # Task2 - Top-10 Best Drivers
    task2 = taxiLinesCorrected.map(lambda x: (x[1],x[16] / (x[4]/60)))
    result2=task2.mapValues(lambda x:(x,1)).reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1])).map(lambda x:(x[0],x[1][0]/x[1][1])).top(10, lambda x: x[1])
    answerFor2 = sc.parallelize(result2)
    answerFor2.coalesce(1).saveAsTextFile(sys.argv[3])
    sc.stop()
