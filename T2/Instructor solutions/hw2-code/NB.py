import numpy as np
import scipy.stats
import torch
from torch import Tensor
from torch.autograd import Variable
import string
import re
import sys
from sklearn.preprocessing import binarize
from utils import *

torch.manual_seed(42)

def main():
	train_iter, val_iter, test_iter, text_field = load_imdb()
	C = 2
	V = len(text_field.vocab)
	pi = Tensor(np.zeros(C))
	theta = Tensor(np.zeros((C,V)))

	feature_counts = np.zeros((C,V))
	label_counts = np.zeros((C,1))
	
	for batch in train_iter:
		x = bag_of_words(batch,text_field)
		y = (batch.label - 1).data.numpy()[0]
		label_counts[y] += 1
		feature_counts[y,:] += np.where(x.data.numpy() > 0,1,0).reshape((V,))

	alpha = 0.5*np.ones((C,1))
	beta = 0.5*np.ones(V)
	pi = (label_counts + alpha) / (label_counts.sum() + alpha.sum())
	theta = (feature_counts + beta[0]) / (label_counts + beta.sum())
	
	correct = 0
	total = 0

	for batch in test_iter:
		x = bag_of_words(batch,text_field)
		y = batch.label - 1
		x = np.where(x.data.numpy() > 0,1,0).reshape((V,))
		preds = np.zeros(C)
		for k in range(C):
			ones = np.dot(x, np.log(theta[k]))
			zeros = np.dot(1-x,np.log(1 - theta[k]))
			preds[k] = ones + zeros + np.log(pi[k])
		y_pred = preds.argmax()
		if y_pred == y.data.numpy():
			correct += 1
		total += 1

	accuracy = correct/float(total)
	print "Test Accuracy:",accuracy

if __name__ == "__main__":
    main()
