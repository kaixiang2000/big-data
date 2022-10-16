from __future__ import print_function
from pyspark import SparkConf,SparkContext
import sys
import re
import numpy as np

from numpy import dot
from numpy.linalg import norm

from operator import add
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext

from pyspark.sql import SparkSession
from pyspark.sql import SQLContext

from pyspark.sql.types import *
from pyspark.sql import functions as func
from pyspark.sql.functions import *
from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.sql.functions import mean, col
from pyspark.sql import functions as F
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import lit
from pyspark.sql import functions as func
from pyspark.sql.types import StringType
from pyspark.sql.functions import mean, col
from pyspark.sql import DataFrameStatFunctions as statFunc

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
if __name__ == "__main__":
    df = sqlContext.read.format('csv').options(header='False', inferSchema='True', sep=","). \
        load(sys.argv[1])
    colnames = ['ID', 'cate']
    df = df.toDF(*colnames)
    #task1 provide summary statistics (max, average, median, std) about the
    #number of Wikipedia categories used for Wikipedia pages.
    df1=df.groupBy("ID").count()
    print(df1.agg(func.max("count")).collect())
    print(df1.agg(func.avg("count")).collect())
    print(df1.agg(func.stddev_pop("count")).collect())
    print(df1.approxQuantile("count",[0.5], 0.25))
    #task2 find the top 10 most used Wikipedia categories
    df2=df.drop("ID").withColumn("count",lit(1))
    df2.groupBy("cate").agg({"count":"sum"}).orderBy("sum(COUNT)", ascending=False).limit(10).show()
