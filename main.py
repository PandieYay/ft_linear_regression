import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(x, y, iterations=1000, learning_rate=0.0001):
	intercept = 0.01
	gradient = 0.1
	m = len(x)

	gradients = []

	def estimatePrice(mileage):
		return intercept + (gradient * mileage)

	def MSE():
		# TODO
		equation = (1 / m) * (estimatePrice(x[1]) - y[1]) ** 2

	for i in range(iterations):
		summationIntercept = 0
		summationGradient = 0

		current_cost = MSE()
		gradients.append(gradient)
		# m will already be - 1 as per formula
		for count in range(m):
			summationIntercept += estimatePrice(x[count]) - y[count]
			summationGradient += (estimatePrice(x[count]) - y[count]) * x[count]

		intercept = learning_rate * (1 / m) * summationIntercept
		gradient = learning_rate * (1 / m) * summationGradient
		# Printing the parameters for each 1000th iteration
		print(
			f"Iteration {i+1}: Gradient \
		{gradient}, Intercept {intercept}"
		)
	# plt.figure(figsize = (8,6))
	# plt.plot(weights, costs)
	# plt.scatter(weights, costs, marker='o', color='red')
	# plt.title("Cost vs Weights")
	# plt.ylabel("Cost")
	# plt.xlabel("Weight")
	# plt.show()


def main():
	mileage = input("Enter your mileage to estimate price: ")
	if not mileage.isdigit():
		print("Please enter a valid number")

	# 1km | 1km = 0.6213 miles | 1 mile = 1.609 km
	myData = pd.read_csv("test.csv")
	m = len(myData)

	# Convert km to miles
	milesArray = myData.km / 1.609

	gradient_descent(milesArray, myData.price)

	# # y=mx+b
	# theta0 = 0
	# theta1 = 0
	# learningRate = 0.1

	# for index in range(10):
	#	 summationTmp0 = 0
	#	 summationTmp1 = 0
	#	 # m will already be - 1 as per formula
	#	 for i in range(m):
	# summationTmp0 += estimatePrice(myData.km[i] / 1.609) - myData.price[i]
	#		 summationTmp1 += (estimatePrice(myData.km[i] / 1.609) - myData.price[i]) * (myData.km[i] / 1.609)
	#	 print("YIPPEE", summationTmp0)
	#	 print("YIPPEE", summationTmp1)
	#	 tmpTheta0 = 0
	#	 tmpTheta1 = 0
	#	 tmpTheta0 = learningRate * (1 / len(myData.km)) * summationTmp0
	#	 tmpTheta1 = learningRate * (1 / len(myData.km)) * summationTmp1
	#	 theta0 = tmpTheta0
	#	 theta1 = tmpTheta1
	#	 print(theta0)
	#	 print(theta1)


if __name__ == "__main__":
	main()
