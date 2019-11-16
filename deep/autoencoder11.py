import numpy as np

def f(w,b,x):
	return 1.0 / (1.0 + np.exp(-(w*x + b)))
	
def grad_w(w,b,x):
	fx = f(w,b,x)
	return fx

def grad_b(w,b,x):
	fx = f(w,b,x)
	return fx
	
def loss(w,b):
	error =0
	
	error += 0.5 * (w - b) ** 2
	return error


parameters1 = np.load('autoencoder1.npy')
w_e_1 = parameters1[0]
b_1 = parameters1[1]
w_d_1 = parameters1[2]
e_1 = parameters1[3]
w = grad_w(b_1,w_e_1,e_1)
#b= grad_b(w_e_1,b_1,w_d_1)

print(b_1)
parameters2 = np.load('autoencoder2.npy')
w_e_2 = parameters2[0]
b_2 = parameters2[1]
w_d_2 = parameters2[2]
e_2 = parameters2[3]