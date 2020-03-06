"""
My take at 
For N = 10
Avg. accuracy:  0.8910620000000005
Avg. itters:    9.663

For N = 100
Avg. accuracy:  0.9865750000000042
Avg. itters:    112.753
"""

from random import random
import numpy as np

def ran_number():
  return random() * 2 - 1

def ran_point():
  return [ran_number() for _ in range(d)]
  
def ran_f():
  #random 2 points
  x1,y1 = ran_point()
  x2,y2 = ran_point()

  #clasical from Ax - y + C = 0, where A is slope
  A = (y1 - y2) / (x1 - x2)
  C = y1 - A * x1
  # print("f(x) = %fx - y + %f = 0"%(A,C))
  def f(x):
    distance = A * x[0] - x[1] + C
    return distance >= 0
  return f

class Perceptron(object):
    def __init__(self, dimensions):
        self.w = np.zeros(dimensions + 1)
           
    def predict(self, x):
        sum = self.w[0] + np.dot(x, self.w[1:])
        if sum > 0:
          activation = 1
        else:
          activation = 0            
        return activation
    def find_missclassified(self, X, Y):
      missclassified = []
      for i, (x, y) in enumerate(zip(X, Y)):
        if self.predict(x) != y:
          missclassified.append(i)
      return missclassified

    def fit(self, X, Y, lr=0.01, max_iter=10000):
      for iterration in range(max_iter):
        miss = self.find_missclassified(X, Y)
        E_in = len(miss)
        if E_in == 0:
          return iterration
        rnd_miss = miss[int(random() * len(miss) - .5)]
        x, y = X[rnd_miss], Y[rnd_miss]
        prediction = self.predict(x)
        self.w[1:] += lr*(y-prediction)*x
        self.w[0] += lr*(y-prediction)


d = 2
N = 10

rnd_points = [ran_point() for _ in range(1000)]
accuracies = []
itters = []

for run in range(1000):
  X=np.array([ran_point() for _ in range(N)])
  f = ran_f()
  Y=np.array([f(x) for x in X])
  preceptron = Perceptron(d)
  itters.append(preceptron.fit(X, Y))

   
  accuracies.append(sum([f(p) == preceptron.predict(p) for p in rnd_points]) / len(rnd_points))

print("For N = ", N)
print("Avg. accuracy: ", sum(accuracies)/len(accuracies))
print("Avg. itters: ", sum(itters)/len(itters))
