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
	error = 0
	for x,y in zip(X,Y):
		fx = f(w,b,x)
		error += 0.5 * (fx - y) ** 2
		return error


filename = 'data.csv'
df = pd.read_csv(filename)
X = df ['x']
Y = df ['y']

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
		
	m_w = beta1 * m_w + (1 - beta1) * dw
	m_b = beta1 * m_b + (1 - beta1) * db
		
	v_w = beta2 * v_w + (1 - beta2) * dw ** 2
	v_b = beta2 * v_b + (1 - beta2) * db ** 2
		
	m_w_hat = m_w / (1 - math.pow(beta1,i+1))
	m_b_hat = m_b / (1 - math.pow(beta1,i+1))
		
	v_w_hat = v_w / (1 - math.pow(beta2,i+1))
	v_b_hat = v_b / (1 - math.pow(beta2,i+1))
		
	w = w - (eta / np.sqrt(v_w_hat + eps))* m_w_hat
	b = b - (eta / np.sqrt(v_b_hat + eps)) * m_b_hat 
	r = loss(w,b)
	print (r)
	losshistory.append(r)
	plt.plot(losshistory)
	
plt.show()

		
			

	


	

	
