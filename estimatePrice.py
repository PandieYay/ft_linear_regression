import pandas as pd
import sys


def estimatePrice(theta0, theta1, mileage):
    return theta0 + (theta1 * mileage)


def main():
    mileage = input("Enter your mileage to estimate price: ")
    try:
        mileage = float(mileage)
    except ValueError:
        sys.exit("Please enter a valid number")

    thetas = pd.read_csv("thetas.csv")
    theta0 = float(thetas.theta0[0])  # Intercept
    theta1 = float(thetas.theta1[0])  # Gradient

    print("Price is:", estimatePrice(theta0, theta1, mileage))


if __name__ == "__main__":
    main()
