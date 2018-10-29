from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import timeit



IT= 10000
K= 20
text = 'Test'+str(IT)


data = load_breast_cancer()

X = data['data']
y = data['target']
X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X)


def logistic_func(theta, x):
    return float(1) / (1 + math.e**(-x.dot(theta)))
def log_gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    final_calc = first_calc.T.dot(x)
    return final_calc
def cost_func(theta, x, y):
    log_func_v = logistic_func(theta,x)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)
def grad_desc(theta_values, X, y, lr=.00001, converge_change=.01):

    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    cost_iter = []
    cost = cost_func(theta_values, X, y)
    cost_iter.append([0, cost])
    change_cost = 1
    i = 1
    while(i<IT):
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y))
        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost	
    	i+=1
    return theta_values, np.array(cost_iter), cost
def pred_values(theta, X, hard=True):
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return pred_value
    return pred_prob

t=[]
l= []
for j in xrange(K):
	
	start = timeit.default_timer()
	shape = X.shape[1]
	start = timeit.default_timer()
	y_flip = np.logical_not(y) 
	betas = np.zeros(shape)
	fitted_values, cost_iter, cost = grad_desc(betas, X, y_flip)
	l.append(cost)
	predicted_y = pred_values(fitted_values, X)
	predicted_y
	np.sum(y_flip == predicted_y)
	stop = timeit.default_timer()
	time = (stop-start)
	t.append(time)

writer = pd.ExcelWriter(text+'.xlsx', engine='xlsxwriter')
data_insertion={'Data':'Breast_cancer','Algorithm':'Logistic Regression','Iterations':IT,'Loss':l,'Time':t}
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Data')
df.mean().to_excel(writer, sheet_name='Mean')
df.var().to_excel(writer, sheet_name='Variance')
writer.save()



print"#Result:"
print " "
print df
print " "
print "Mean:"
print " "
print df.mean()
print " "
print "Variance:"
print " "
print df.var()






