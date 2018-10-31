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
init_w,init_b = -2,-2
w,b,eta,max_epochs= init_w,init_b,1.0,100
prev_v_w,prev_v_b,gamma=0,0,0.9
losshistory = []
for i in range(max_epochs):
	dw,db=0,0
	v_w = gamma * prev_v_w
	v_b = gamma * prev_v_b
	for x,y in zip(X,Y):
		dw += grad_w(prev_v_w,b,x,y)
		db += grad_b(prev_v_b,b,x,y)
	v_w = gamma * prev_v_w + eta * dw
	v_b = gamma * prev_v_b + eta * db
	w = w - v_w
	b = b - v_b
	prev_v_w =v_w
	prev_v_b = v_b
	r = loss(w,b)
	print (r)
	losshistory.append(r)
	plt.plot(losshistory)
	
plt.show()
