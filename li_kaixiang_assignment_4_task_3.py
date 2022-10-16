from __future__ import print_function

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
	corpus = sc.textFile(sys.argv[1], 1)
	testCorpus=sc.textFile(sys.argv[2], 1)
	keyAndText = corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:][:-6]))
	keyAndTextTest = testCorpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')],x[x.index('">') + 2:][:-6]))
	regex = re.compile('[^a-zA-Z]')

	keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	keyAndListOfWordsTest = keyAndTextTest.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))
	task1=keyAndListOfWords.flatMap(lambda x:x[1]).map(lambda x:(x,1)).reduceByKey(lambda x,y:x+y)
	mostFrequence=task1.top(10000,lambda x: x[1])
	topWordsK=sc.parallelize(range(10000))
	dictionary = topWordsK.map (lambda x : (mostFrequence[x][0], x))
	allWordsWithDocID = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))
	allWordsWithDocIDTest = keyAndListOfWordsTest.flatMap(lambda x: ((j, x[0]) for j in x[1]))
	#only tf 
	allDictionaryWords = dictionary.join(allWordsWithDocID)
	allDictionaryWordsTest = dictionary.join(allWordsWithDocIDTest)
	justDocAndPos = allDictionaryWords.map(lambda x: (x[1][1], x[1][0]))
	justDocAndPosTest = allDictionaryWordsTest.map(lambda x: (x[1][1], x[1][0]))
	allDictionaryWordsInEachDoc = justDocAndPos.groupByKey()
	allDictionaryWordsInEachDocTest = justDocAndPosTest.groupByKey()
	allDocsAsNumpyArrays = allDictionaryWordsInEachDoc.map(lambda x: (x[0], freqArray(x[1])))
	allDocsAsNumpyArraysTest  = allDictionaryWordsInEachDocTest.map(lambda x: (x[0], freqArray(x[1])))
	testdata=allDocsAsNumpyArraysTest.map(lambda x:(1 if 'AU' in x[0] else 0, x[1]))
	task2=allDocsAsNumpyArrays.map(lambda x:(1 if 'AU' in x[0] else 0, x[1]))
	testdata1 = testdata.map(lambda x: (x[0],np.append(x[1],1)))
	traindata1=task2.map(lambda x: (x[0],np.append(x[1],1)))
	traindata1.cache()
	testdata1.cache()
	train_size = task2.count()
	n_samples1=traindata1.filter(lambda x: x[0]==1).count()
	n_samples0=traindata1.filter(lambda x: x[0]==0).count()
	n_samples = [n_samples0, n_samples1]
	w0=train_size/(2*n_samples[0])
	w1 = train_size/(2*n_samples[1])
	#function optimizer
	# Training model
	# The optimised version of the code
	def LogisticRegression_weighted(traindata=traindata1,
						max_iteration = 50,
						learningRate = 0.01,
						regularization = 0.01,
						mini_batch_size = 512,
						tolerance = 10e-8,
						beta = 0.9,
						beta2 = 0.999,
						optimizer = 'SGD',  #optimizer: 'Momentum' / 'Adam' / 'Nesterov' / 'Adagrad' / 'RMSprop' / 'SGD' 
						train_size=1
						):

		# initialization
		prev_cost = 0
		L_cost = []
		prev_validation = 0
		
		parameter_size = len(traindata.take(1)[0][1])
		np.random.seed(0)
		parameter_vector = np.random.normal(0, 0.1, parameter_size)
		momentum = np.zeros(parameter_size)
		prev_mom = np.zeros(parameter_size)
		second_mom = np.array(parameter_size)
		gti = np.zeros(parameter_size)
		epsilon = 10e-8
		
		for i in range(max_iteration):

			bc_weights = parameter_vector
			min_batch = traindata.sample(False, mini_batch_size / train_size, 1 + i)
	
			# treeAggregate(vector of gradients, total_cost, number_of_samples)
			# Calcualtion of positive class. Only the samples labeled as 1 are filtered and then  processed
			res1 = min_batch.filter(lambda x: x[0]==1).treeAggregate(
				(np.zeros(parameter_size), 0, 0),
				lambda x, y:(x[0]+\
							(y[1])*(-y[0]+(1/(np.exp(-np.dot(y[1], bc_weights))+1))),\
							x[1]+\
							y[0]*(-(np.dot(y[1], bc_weights)))+np.log(1 + np.exp(np.dot(y[1],bc_weights))),\
							x[2] + 1),
				lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2])
				)        
			# Calcualtion of negative class. Only the samples labeled as 0 are filtered and then processed
			res0 = min_batch.filter(lambda x: x[0]==0).treeAggregate(
				(np.zeros(parameter_size), 0, 0),
				lambda x, y:(x[0]+\
							(y[1])*(-y[0]+(1/(np.exp(-np.dot(y[1], bc_weights))+1))),\
							x[1]+\
							y[0]*(-(np.dot(y[1], bc_weights)))+np.log(1 + np.exp(np.dot(y[1],bc_weights))),\
							x[2] + 1),
				lambda x, y:(x[0] + y[0], x[1] + y[1], x[2] + y[2])
				)        
			
			# The total gradients are a weighted sum
			gradients = w0*res0[0] + w1*res1[0]
			sum_cost = w0*res0[1] + w1*res1[1]
			num_samples = res0[2] + res1[2]
			
			cost =  sum_cost/num_samples + regularization * (np.square(parameter_vector).sum())

			# calculate gradients
			gradient_derivative = (1.0 / num_samples) * gradients + 2 * regularization * parameter_vector
			
			if optimizer == 'SGD':
				parameter_vector = parameter_vector - learningRate * gradient_derivative

			if optimizer =='Momentum':
				momentum = beta * momentum + learningRate * gradient_derivative
				parameter_vector = parameter_vector - momentum
				
			if optimizer == 'Nesterov':
				parameter_temp = parameter_vector - beta * prev_mom
				parameter_vector = parameter_temp - learningRate * gradient_derivative
				prev_mom = momentum
				momentum = beta * momentum + learningRate * gradient_derivative
				
			if optimizer == 'Adam':
				momentum = beta * momentum + (1 - beta) * gradient_derivative
				second_mom = beta2 * second_mom + (1 - beta2) * (gradient_derivative**2)
				momentum_ = momentum / (1 - beta**(i + 1))
				second_mom_ = second_mom / (1 - beta2**(i + 1))
				parameter_vector = parameter_vector - learningRate * momentum_ / (np.sqrt(second_mom_) + epsilon)

			if optimizer == 'Adagrad':
				gti += gradient_derivative**2
				adj_grad = gradient_derivative / (np.sqrt(gti)  + epsilon)
				parameter_vector = parameter_vector - learningRate  * adj_grad
			
			if optimizer == 'RMSprop':
				sq_grad = gradient_derivative**2
				exp_grad = beta * gti / (i + 1) + (1 - beta) * sq_grad
				parameter_vector = parameter_vector - learningRate / np.sqrt(exp_grad + epsilon) * gradient_derivative
				gti += sq_grad
				
				
			print("Iteration No.", i)
			
			# Stop if the cost is not descreasing
			if abs(cost - prev_cost) < tolerance:
				print("cost - prev_cost: " + str(cost - prev_cost))
				break
			prev_cost = cost
			L_cost.append(cost)
			
		return parameter_vector, L_cost

	# Fit Train dataset
	parameter_vector_sgd_w, L_cost_sgd_w = LogisticRegression_weighted(traindata=traindata1,
					max_iteration = 50,
					learningRate = 0.05,
					regularization = 0.05,
					mini_batch_size = 512,
					tolerance = 10e-8,
					optimizer = 'RMSprop',
					train_size = train_size
                      )	
	# Create an RDD wiht the true value and the predicted value (true, predicted)
	predictions = testdata1.map(lambda x: (x[0], 1 if np.dot(x[1],parameter_vector_sgd_w)>0 else 0))

	true_positive = predictions.map(lambda x: 1 if (x[0]== 1) and (x[1]==1) else 0).reduce(lambda x,y:x+y)
	false_positive = predictions.map(lambda x: 1 if (x[0]== 0) and (x[1]==1) else 0).reduce(lambda x,y:x+y)

	true_negative = predictions.map(lambda x: 1 if (x[0]== 0) and (x[1]==0) else 0).reduce(lambda x,y:x+y)
	false_negative = predictions.map(lambda x: 1 if (x[0]== 1) and (x[1]==0) else 0).reduce(lambda x,y:x+y)

	# Print the Contingency matrix
	print("--Contingency matrix--")
	print(f" TP:{true_positive:6}  FP:{false_positive:6}")
	print(f" FN:{false_negative:6}  TN:{true_negative:6}")
	print("----------------------")

	# Calculate the Accuracy and the F1
	test_num = testdata1.count()
	accuracy = (true_positive+true_negative)/(test_num)
	f1 = true_positive/(true_positive+0.5*(false_positive+false_negative))
	print(f"Accuracy = {accuracy}  \nF1 = {f1}")
	sc.stop()