import csv
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
	for x,y in zip(X,Y):
		fx = f(w,b,x)
		error = 0.5 * (fx - y) ** 2
		return error





X = [0,-9,-14,-10,-6,12,19,-18,7,-4,17,-1,1,8,-13,-20,-12,-8,-15,14,4,-2,11,-16,-3,10,3,-11,9,15,-7,5,-19,2,18,13,16,-17,-5,6,]
Y = [0.731058579,4.14E-08,1.88E-12,5.60E-09,1.67E-05,1,1,6.31E-16,0.999999694,0.000911051,1,0.268941421,0.952574127,0.999999959,1.39E-11,1.15E-17,1.03E-10,3.06E-07,2.54E-13,1,0.999876605,0.047425873,1,3.44E-14,0.006692851,0.999999999,0.999088949,7.58E-10,0.999999994,1,2.26E-06,0.999983299,8.53E-17,0.993307149,1,1,1,4.66E-15,0.000123395,0.99999774]
#X = [0.5,2.5]
#Y = [0.2,0.9]

init_w,init_b=1,1
w_b_dw_db = [(init_w,init_b,0,0)]
w_history,b_history,error_history,losshistory=[],[],[],[]
w,b,eta,mini_batch_size,num_points_seen = init_w,init_b,0.01,10,0 
m_w,m_b,v_w,v_b,eps,beta1,beta2,max_epochs=0,0,0,0,1e-8,0.9,0.99,1000

for i in range (max_epochs):
	dw,db = 0,0
	for x,y in zip(X,Y):
		dw+= grad_w(w,b,x,y)
		db+= grad_b(w,b,x,y)
		num_points_seen += 1
		
	if num_points_seen % mini_batch_size == 0:
		
		m_w = beta1 * m_w + (1 - beta1) * dw
		m_b = beta1 * m_b + (1 - beta1) * db
		
		v_w = beta2 * v_w + (1 - beta2) * dw ** 2
		v_b = beta2 * v_b + (1 - beta2) * db ** 2
		
		m_w = m_w / (1 - math.pow(beta1,i+1))
		m_b = m_b / (1 - math.pow(beta1,i+1))
		
		v_w = v_w / (1 - math.pow(beta2,i+1))
		v_b = v_b / (1 - math.pow(beta2,i+1))
		
		w = w - (eta / np.sqrt(v_w + eps))* m_w
		b = b - (eta / np.sqrt(v_b + eps)) * m_b 
	r = loss(w,b)
	print (r)
	losshistory.append(r)
	plt.plot(losshistory)
	
plt.show()

		
			

	


	

	
