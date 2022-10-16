from __future__ import print_function
import findspark
findspark.init()
import re
import sys
import numpy as np
from operator import add

from pyspark import SparkContext
numTopWords = 10000
def freqArray (listOfIndices):
	global numTopWords
	returnVal = np.zeros (numTopWords)
	for index in listOfIndices:
		returnVal[index] = returnVal[index] + 1
	mysum = np.sum (returnVal)
	returnVal = np.divide(returnVal, mysum)
	return returnVal


if __name__ == "__main__":

	sc = SparkContext(appName="Assignment-4")

	### Task 1
	### Data Preparation
	corpus = sc.textFile(sys.argv[1],1)
	keyAndText = corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')

	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	#[('AU35',['consideration','of',...])] to (word,1) then reduce get numbers of words
	task1=keyAndListOfWords.flatMap(lambda x:x[1]).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y)
	#top 10000 most frequence word list to rdd
	mostFrequence=task1.top(10000,lambda x: x[1])
	#change frequence to 0...10000
	topWordsK=sc.parallelize(range(10000))
	dictionary = topWordsK.map (lambda x : (mostFrequence[x][0], x))
	dictionary.cache()
	### Include the following results in your report:
	print("Index for 'applicant' is",dictionary.filter(lambda x: x[0]=='applicant').take(1)[0][1])
	print("Index for 'and' is",dictionary.filter(lambda x: x[0]=='and').take(1)[0][1])
	print("Index for 'attack' is",dictionary.filter(lambda x: x[0]=='attack').take(1)[0][1])
	print("Index for 'protein' is",dictionary.filter(lambda x: x[0]=='protein').take(1)[0][1])
	print("Index for 'car' is",dictionary.filter(lambda x: x[0]=='car').take(1)[0][1])
	print("Index for 'in' is",dictionary.filter(lambda x: x[0]=='in').take(1)[0][1])
