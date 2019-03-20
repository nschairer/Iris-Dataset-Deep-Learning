from sklearn import datasets
import numpy as np

def normalizelabels(l):
	f = []
	for x in l:
		if x == 0:
			f.append([1,0,0])
		elif x == 1:
			f.append([0,1,0])
		else:
			f.append([0,0,1])
	L = np.array(f)
	return L

def sigmoid(x):
	return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
	return x * (1.0 - x)

def softmax(x, axis=-1):
	e_x = np.exp(x - np.max(x)) # same code
	return e_x / e_x.sum(axis=axis, keepdims=True)
def soft_max_derivative(p):
	return p*(1-p)

def cross_entropy(output, labels):
	#sum of total error 
	loss = -(labels*np.log(output))
	loss_percentage = np.sum(loss)/(output.shape[0]*output.shape[1])
	print('Loss: {}%'.format(loss_percentage*100))

def cross_entropy_derivative(p,y):
	return (p - y)


class basicNeuralNetwork():
	def __init__(self, inputs,labels):
		self.input = inputs
		self.weights1 = np.random.rand(self.input.shape[1], 4)
		self.weights2 = np.random.rand(4, 4)
		self.weights3 = np.random.rand(4, 3)
		self.labels = labels
		self.output = np.zeros(self.input.shape[1])

	def forwardPropagate(self):
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
		self.output = softmax(np.dot(self.layer2, self.weights3))

	def calculateLoss(self):
		cross_entropy(self.output, self.labels)

	def backwardPropagation(self):
		part1 = soft_max_derivative(self.output)*cross_entropy_derivative(self.output,self.labels)
		delta_weights3 = np.dot(self.layer2.T, part1)
		d_weights2 = np.dot(self.layer1.T,  (np.dot(part1,self.weights3.T) * sigmoid_derivative(self.layer2)))
		part2 = np.dot(part1,self.weights3.T) * sigmoid_derivative(self.layer2)
		d_weights1 = np.dot(self.input.T, (np.dot(part2,self.weights2.T)*sigmoid_derivative(self.layer1)))
		self.weights3 -= delta_weights3 *0.0032
		self.weights2 -= d_weights2*0.0032
		self.weights1 -= d_weights1*0.0032

	# def guess(self, guess_val):
	# 	z1 = 
	# 	return 

	def train(self,epochs):
		epoch = 0
		for x in range(epochs):
			self.forwardPropagate()
			if epoch == 100:
				self.calculateLoss()
				epoch = 0
			self.backwardPropagation()
			epoch+=1


if __name__ == '__main__':
	np.random.seed(1)
	iris = datasets.load_iris()
	inputs = iris['data']
	labels = iris['target']
	bNN = basicNeuralNetwork(inputs, normalizelabels(labels))
	bNN.train(20000)
	for x in bNN.output.tolist():
		m = x.index(max(x))
		if(m == 0):
			print('Setosa')
		elif(m==1):
			print('Versicolor')
		elif(m==2):
			print('Verginica')

