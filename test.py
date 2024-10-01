import pandas as pd
import numpy as np

def example_gradient_descent(x, y, iterations = 1000, learning_rate = 0.0001):
	# Initializing weight, bias, learning rate and iterations
	current_weight = 0.1
	current_bias = 0.01
	n = float(len(x))
	
	costs = []
	weights = []
	previous_cost = None
	
	# Estimation of optimal parameters 
	# for i in range(iterations):
    # Making predictions
	y_predicted = (current_weight * x) + current_bias

	# Calculating the gradients
	weight_derivative = -(1/n) * sum(x * (y-y_predicted))
	bias_derivative = -(1/n) * sum(y-y_predicted)
	
	
	current_bias = current_bias - (learning_rate * bias_derivative)
	print(bias_derivative)

def gradient_descent(x, y, iterations = 1000, learning_rate = 0.0001):
	gradient = 0.1
	intercept = 0.01
	m = len(x)

	def estimatePrice(mileage):
		return (intercept + (gradient * mileage))

	# for i in range(iterations):
	y_predicted = (gradient * x) + intercept
	summationTmp0 = 0
	summationTmp1 = 0
	# m will already be - 1 as per formula
	for i in range (m):
		summationTmp0 += estimatePrice(x[i]) - y[i]
		summationTmp1 += (estimatePrice(x[i]) - y[i]) * x[i]
	tmpTheta0 = (1 / m) * summationTmp0
	tmpTheta1 = (1 / m) * summationTmp1
	print(tmpTheta0)

def main():    
	mileage = input("Enter your mileage to estimate price: ")
	if (not mileage.isdigit()):
		print("Please enter a valid number")

	# 1km | 1km = 0.6213 miles | 1 mile = 1.609 km
	myData = pd .read_csv('test.csv')
	m = len(myData)

	X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
		55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
		45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
		48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
	Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
		78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
		55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
		60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

	gradient_descent(X, Y)
	example_gradient_descent(X, Y)

if __name__=="__main__":
	main()