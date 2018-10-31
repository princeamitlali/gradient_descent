import csv
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import style
def f(w,b,x):
	return 1.0 / (1.0 + np.exp(-(w*x + b)))
	
def grad_w(w,b,x,y):
	fx = f(w,b,x)
	return (fx - y ) * fx * (1 - fx) * x

def grad_b(w,b,x,y):
	fx = f(w,b,x)
	return (fx - y ) * fx * (1 - fx)
	
def loss(w,b):
	error =0
	for x,y in zip(X,Y):
		fx = f(w,b,x)
		error += 0.5 * (fx - y) ** 2
		return error
		
filename = 'data.csv'
df = pd.read_csv(filename)
X = df ['x']
Y = df ['y']
losshistory = []

w,b,eta,max_epochs= -2,-2,1.0,100
for i in range(max_epochs):
	dw,db=0,0
	for x,y in zip(X,Y):
		dw += grad_w(w,b,x,y)
		db += grad_b(w,b,x,y)
	w = w - eta * dw
	b = b - eta * dw
	r = loss(w,b)
	print (r)
	losshistory.append(r)
	plt.plot(losshistory)
	

plt.show()
	
