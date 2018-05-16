import matplotlib.pyplot as plt 
import numpy as np 

def misplot(truths, predictions, for_class, labels):
	n_classes = truths.shape[1]

	for i in range(n_classes):
		print(np.argmax(truths, axis=1))
		class_preds = predictions[np.argmax(truths, axis=1) == i]
		N = len(class_preds)

		xs = np.sort(class_preds[:, for_class])
		ys = 1-np.arange(N, dtype=np.float)/N 

		#if i != for_class:
		#	ys = 1 - ys
		
		plt.semilogy(xs, ys, label=labels[i])

	plt.grid(True, which="both")
	plt.legend()
	plt.xlabel("Cut probability")
	plt.ylabel("Cumulative fraction of events that survive cut")
	plt.show()

def rocplot(truths, predictions, for_class):
	true_classes = np.argmax(truths, axis=1)
	true_prs = predictions[true_classes == for_class]
	false_prs = predictions[true_classes != for_class]

	xs = []
	ys = []
	for i in np.linspace(0,1,100):
		xs.append(np.sum(false_prs[:, for_class] > i)/len(false_prs))
		ys.append(np.sum(true_prs[:, for_class] > i)/len(true_prs))

	plt.plot(xs, ys)
	plt.show()

def poiss(x, mean):
	import scipy.special
	#print(x)
	#print(mean)
	return (mean**x)*np.exp(-mean)/scipy.special.gamma(x+1)

def sigplot(truths, predictions, weights, for_class, labels):
	true_classes = np.argmax(truths, axis=1)
	print(np.sum(weights))
	true_prs = predictions[true_classes == for_class]
	true_weights = weights[true_classes == for_class]
	false_prs = predictions[true_classes != for_class]
	false_weights = 5*weights[true_classes != for_class]

	bgs = []
	sigs = []
	cuts = np.linspace(0,1,100,endpoint=False)
	#print(false_weights.shape)
	for i in cuts:
		bgs.append(np.sum(false_weights*(false_prs[:, for_class] > i)))
		sigs.append(np.sum(true_weights*(true_prs[:, for_class] > i)))

	plt.semilogy(cuts, sigs, label=labels[for_class])
	plt.semilogy(cuts, bgs, label="NC")
	#plt.semilogy(cuts, np.array(sigs)/np.array(bgs))
	plt.legend()
	plt.grid(True, which="both")
	plt.show()
	plt.semilogy(cuts, poiss(x=5*np.array(sigs)+5*np.array(bgs), mean=5*np.array(bgs)))
	plt.show()

if __name__ == '__main__':
	import pickle
	import sys

	with open(sys.argv[1], 'rb') as infile:
		data = pickle.load(infile)

	#misplot(data['truths'], data['predictions'], for_class=0, labels=data['label_names'])
	#rocplot(data['truths'], data['predictions'], for_class=0)
	sigplot(data['truths'], data['predictions'], data['weights'], for_class=0, labels=data['label_names'])

