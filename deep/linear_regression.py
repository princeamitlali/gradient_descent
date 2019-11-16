#import pandas as pd
import quandl,math,datetime
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn import preprocessing,svm
#from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt
#from matplotlib import style
#import pickle
#style.use('ggplot')
key = "zWKyUkJFGY34G5LFbFTZ"
df = quandl.get("WIKI/KO",trim_start = "2000-12-12", trim_end = "2014-12-30",authoken=key)
#df = df[['date','open','high','low','close','last','volume','intrest']]
print(df.head)