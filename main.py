import pandas as pd

mileage = input("Enter your mileage to estimate price: ")
if (not mileage.isdigit()):
    print("Please enter a valid number")

# 1km / 1.609 = 0.621371 miles; 1 mile = 1.609 km
df = pd.read_csv('data.csv')
print(df.km)