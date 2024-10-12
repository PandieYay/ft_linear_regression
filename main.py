import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def z_score_normalization(x, y):
	totalx = 0
	totaly = 0

	# calculate mean
	for i in range(len(x)):
		totalx += x[i]
	for i in range(len(y)):
		totaly += y[i]
	meanx = totalx / len(x)
	meany = totaly / len(y)

	totalx = 0
	totaly = 0
	# calculate standard deviation
	for i in range (len(x)):
		totalx += (x[i] - meanx)**2
	for i in range(len(y)):
		totaly += (y[i] - meany	)**2
	deviationx = math.sqrt(totalx / len(x))
	deviationy = math.sqrt(totaly / len(y))

	z_score_x = (x - meanx) /  deviationx
	z_score_y = (y - meany) / deviationy

	return (z_score_x, z_score_y, meany, deviationy)

# Gradient is weight, bias is intercept
def gradient_descent(x, y, iterations=1000, learning_rate=0.01, stopping_threshold = 1e-6):
	intercept = 0.01
	gradient = 0.1
	m = len(x)

	costs = []
	gradients = []
	previous_cost = None

	def estimatePrice(mileage):
		return intercept + (gradient * mileage)

	def MSE():
		summationMSE = 0
		for index in range(m):
			summationMSE += (estimatePrice(x[index]) - y[index])**2
		cost = (1 / m) * summationMSE
		return cost

	for i in range(iterations):
		summationIntercept = 0
		summationGradient = 0

		# m will already be - 1 as per formula
		for index in range(m):
			summationIntercept += estimatePrice(x[index]) - y[index]
			summationGradient += (estimatePrice(x[index]) - y[index]) * x[index]
		current_cost = MSE()

		if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
			break
		previous_cost = current_cost
		costs.append(current_cost)
		gradients.append(gradient)
		intercept -= learning_rate * (1 / m) * summationIntercept
		gradient -= learning_rate * (1 / m) * summationGradient
		# Printing the parameters for each 1000th iteration
		print(
			f"Iteration {i+1}: Cost {current_cost}, Gradient \
		{gradient}, Intercept {intercept}"
		)
	plt.figure(figsize = (8,6))
	plt.plot(gradients, costs)
	plt.scatter(gradients, costs, marker='o', color='red')
	plt.title("Cost vs Weights")
	plt.ylabel("Cost")
	plt.xlabel("Weight")
	plt.show()
	return gradient, intercept


def main():
	# mileage = input("Enter your mileage to estimate price: ")
	# if not mileage.isdigit():
	# 	print("Please enter a valid number")

	# 1km | 1km = 0.6213 miles | 1 mile = 1.609 km
	myData = pd.read_csv("data.csv")
	m = len(myData)

	# Convert km to miles
	milesArray = myData.km / 1.609

	# X = np.array([32.50234527, 53.42680403, 61.53035803, 47.47563963, 59.81320787,
	# 		55.14218841, 52.21179669, 39.29956669, 48.10504169, 52.55001444,
	# 		45.41973014, 54.35163488, 44.1640495 , 58.16847072, 56.72720806,
	# 		48.95588857, 44.68719623, 60.29732685, 45.61864377, 38.81681754])
	# Y = np.array([31.70700585, 68.77759598, 62.5623823 , 71.54663223, 87.23092513,
	# 		78.21151827, 79.64197305, 59.17148932, 75.3312423 , 71.30087989,
	# 		55.16567715, 82.47884676, 62.00892325, 75.39287043, 81.43619216,
	# 		60.72360244, 82.89250373, 97.37989686, 48.84715332, 56.87721319])

	# X = np.array([1, 2, 3])
	# Y = np.array([100, 200, 300])

	X = np.array(myData.price)
	Y = np.array(milesArray)

	z_score_x, z_score_y, meany, deviationy = z_score_normalization(X, Y)
	estimated_weight, estimated_bias = gradient_descent(z_score_x, z_score_y)
	print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

	# Calculate predictions in normalized space
	Y_pred_normalized = estimated_weight * z_score_x + estimated_bias

	# Reverse the normalization for Y predictions
	Y_pred = Y_pred_normalized * deviationy + meany

	# Plotting the regression line
	plt.figure(figsize = (8,6))
	plt.scatter(X, Y, marker='o', color='red')
	plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red',
			markersize=10,linestyle='dashed')
	plt.xlabel("Price")
	plt.ylabel("Miles")
	plt.show()

if __name__ == "__main__":
	main()
