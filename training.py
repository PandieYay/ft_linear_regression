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
    SST = totaly
    deviationx = math.sqrt(totalx / len(x))
    deviationy = math.sqrt(totaly / len(y))

    z_score_x = (x - meanx) /  deviationx
    z_score_y = (y - meany) / deviationy

    return (z_score_x, z_score_y, meany, deviationy, meanx, deviationx, SST)

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
    # 1km | 1km = 0.6213 miles | 1 mile = 1.609 km
    myData = pd.read_csv("data.csv")
    m = len(myData)

    # Convert km to miles

    X = np.array(myData.km)
    Y = np.array(myData.price)

    # X = np.array(myData.price)
    # Y = np.array(myData.km)

    z_score_x, z_score_y, meany, deviationy, meanx, deviationx, SST = z_score_normalization(X, Y)
    estimated_weight, estimated_bias = gradient_descent(z_score_x, z_score_y)
    print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")

    # Calculate predictions in normalized space
    # Y_pred_normalized = estimated_weight * z_score_x + estimated_bias
    # Reverse the normalization for Y predictionsx
    # Y_pred = Y_pred_normalized * deviationy + meany

    original_weight = estimated_weight * (deviationy / deviationx)
    original_bias = estimated_bias + meany - (original_weight * meanx)

    Y_pred = original_weight * X + original_bias

    thetas = pd.read_csv("thetas.csv")
    thetas.theta0[0] = str(original_bias) # Intercept
    thetas.theta1[0] = str(original_weight) # Gradient
    print(original_bias, original_weight)
    thetas.to_csv("thetas.csv", index=False)

    # Plotting the regression line
    x_min = min(X)
    x_max = max(X)
    y_min = Y_pred[np.where(X == x_min)]
    y_max = Y_pred[np.where(X == x_max)]
    plt.figure(figsize = (8,6))
    plt.scatter(X, Y, marker='o', color='red')
    # plt.scatter(X, Y_pred, color='blue', marker='o')
    plt.plot([x_min, x_max], [y_min, y_max], color='blue',markerfacecolor='red',
            markersize=10,linestyle='dashed')
    plt.xlabel("KM")
    plt.ylabel("Price")
    plt.show()

    # R-Squared: 1-(SSR/SST)
    SSR = 0
    for i in range(len(Y)):
        SSR += (Y[i] - Y_pred[i])**2
    print("MY R2 SCORE IS", 1 - (SSR/SST))


if __name__ == "__main__":
	main()
