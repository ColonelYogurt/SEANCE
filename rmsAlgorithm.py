import math

numbers = [1, 2, 3, 4, 5]

length = len(numbers)
sumOfNumbers = 0

for number in numbers:
    number = pow(number, 2)
    sumOfNumbers = sumOfNumbers + number
    
mean = sumOfNumbers / length
rootMeanSquare = math.sqrt(mean)
print(rootMeanSquare)

