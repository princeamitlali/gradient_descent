import csv
import pandas as pd
import numpy as np
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
init_w,init_b = -2,-2
w,b,etas,max_epochs= init_w,init_b,[0.1,0.5,1.0,5.0,10.0],200
tem_w,tem_b=0,0
losshistory = []
for i in range(max_epochs):
	dw,db=0,0
	for x,y in zip(X,Y):
		dw += grad_w(w,b,x,y)
		db += grad_b(w,b,x,y)
	min_error = 100000
	best_w,best_b = w,b
	for eta in etas:
		tem_w = w - eta*dw
		tem_b = b - eta*db
		if loss(tem_w,tem_b)<min_error:
			best_w = tem_w
			best_b = tem_b
			min_error = loss(tem_w,tem_b)
	w,b= best_w,best_b
	r = loss(w,b)
	print (r)
	losshistory.append(r)
	plt.plot(losshistory)
	
plt.show()
