import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Gradient is weight, bias is intercept
def gradient_descent(x, y, iterations=1000, learning_rate=0.000001, stopping_threshold = 1e-6):
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
		for count in range(m):
			summationMSE += (estimatePrice(x[count]) - y[count])**2
		cost = (1 / m) * summationMSE
		return cost

	for i in range(iterations):
		summationIntercept = 0
		summationGradient = 0

		# m will already be - 1 as per formula
		for count in range(m):
			summationIntercept += estimatePrice(x[count]) - y[count]
			summationGradient += (estimatePrice(x[count]) - y[count]) * x[count]
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
	mileage = input("Enter your mileage to estimate price: ")
	if not mileage.isdigit():
		print("Please enter a valid number")

	# 1km | 1km = 0.6213 miles | 1 mile = 1.609 km
	myData = pd.read_csv("test.csv")
	m = len(myData)

	# Convert km to miles
	milesArray = myData.km / 1.609

	X = np.array([50, 100, 200])
	Y = np.array([50, 100, 200])

	estimated_weight, estimated_bias = gradient_descent(X, Y)
	print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

	Y_pred = estimated_weight*X + estimated_bias

	# Plotting the regression line
	plt.figure(figsize = (8,6))
	plt.scatter(X, Y, marker='o', color='red')
	plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red',
			markersize=10,linestyle='dashed')
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.show()
if __name__ == "__main__":
	main()
