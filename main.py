import pandas as pd

def estimatePrice(mileage):
    return (theta0 + (theta1 * mileage))

mileage = input("Enter your mileage to estimate price: ")
if (not mileage.isdigit()):
    print("Please enter a valid number")

# 1km | 1km = 0.6213 miles | 1 mile = 1.609 km
myData = pd .read_csv('test.csv')
m = len(myData)

# y=mx+b
theta0 = 0
theta1 = 0
learningRate = 0.1
 

for index in range(10):
    summationTmp0 = 0
    summationTmp1 = 0
    # m will already be - 1 as per formula
    for i in range(m):
        summationTmp0 += estimatePrice(myData.km[i] / 1.609) - myData.price[i]
        summationTmp1 += (estimatePrice(myData.km[i] / 1.609) - myData.price[i]) * (myData.km[i] / 1.609)
    print("YIPPEE", summationTmp0)
    print("YIPPEE", summationTmp1)
    tmpTheta0 = 0
    tmpTheta1 = 0
    tmpTheta0 = learningRate * (1 / len(myData.km)) * summationTmp0
    tmpTheta1 = learningRate * (1 / len(myData.km)) * summationTmp1
    theta0 = tmpTheta0
    theta1 = tmpTheta1
    print(theta0)
    print(theta1)