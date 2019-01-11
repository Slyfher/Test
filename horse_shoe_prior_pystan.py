import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import pystan
import datetime
import pandas as pd

diabetes = datasets.load_diabetes()
min_max_scaler = MinMaxScaler()#establece un minimo y max de valores de muestra
diabetes_X = diabetes.data
diabetes_X = min_max_scaler.fit_transform(diabetes_X)#ajuste de data
#diabetes_X = diabetes.data[:, np.newaxis, 2] #take 1

# Split the data into training/testing sets
X = diabetes_X[:-20]
X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
y = diabetes.target[:-20]
y_test = diabetes.target[-20:]

#Horseshoe-prior stan model
model_code = """
data {
  int<lower=0> n;
  int<lower=0> p;
  matrix[n,p] X;
  vector[n] y;
}
parameters {
  vector[p] beta;
  vector<lower=0>[p] lambda;
  real<lower=0> tau;
  real<lower=0> sigma;
}
model {
  lambda ~ cauchy(0, 1);
  tau ~ cauchy(0, 1);
  for (i in 1:p)
    beta[i] ~ normal(0, lambda[i] * tau);
  sigma ~ gamma(0.01,0.01);
  y ~ normal(X * beta, sigma);
}
"""

n, p = X.shape 
data = dict(n=n, p=p, X=X, y=y)

iterations=500 #iterations of algorithm

print " ------------------------------------------------------"
print "| Running HS-prior with: HMC & NUTS | Iterations: ",iterations, " |"
print " ------------------------------------------------------"

fit_nuts = pystan.stan(model_code=model_code, data=data, seed=5, iter=iterations, algorithm="NUTS")

init_hmc = datetime.datetime.now()
fit_hmc = pystan.stan(model_code=model_code, data=data, seed=5, iter=iterations, algorithm="HMC")
#fit_fixed_param = pystan.stan(model_code=model_code, data=data, seed=5, iter=iterations, algorithm="Fixed_param")

#print "FIT MODEL:",fit
#beta = np.mean(fit.extract()['beta'], axis=0)
#ypred = np.dot(X_test, beta)
#print "ypred:",ypred

# Horseshoe, mean squared error: 0.46
#print "hs: ",(np.sum((y_test - ypred)**2) / len(y_test))

#to extract samples,return an array of three dimensions: iterations, chains, parameters
#a = fit_nuts.extract(permuted=False)


#fit_nuts.plot('beta')

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

ax1.violinplot(fit_nuts.extract()['beta'], points=100, vert=False, widths=0.7, showmeans=True, showextrema=True, showmedians=True)
ax1.set_title('NUTS')
ax2.violinplot(fit_hmc.extract()['beta'], points=100, vert=False, widths=0.7, showmeans=True, showextrema=True, showmedians=True)
ax2.set_title('HMC')
#ax3.violinplot(fit_fixed_param.extract()['beta'], points=100, vert=False, widths=0.7, showmeans=True, showextrema=True, showmedians=True)
#ax3.set_title('FIXED_PARAM')
plt.show()

import statsmodels.api as sm
from pandas import Series

#f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
##series1=Series(np.log(fit_nuts.extract()['beta'][:,0])) #desde primera a ultima fila de la columna 0
#series2=Series(np.log(fit_nuts.extract()['beta'][:,1]))
#plt.plot_acf(x=series1,lags=20)
#ax1.plot_acf(x=series2,lags=20)
#plt.show()


##fig = plt.figure(figsize=(12,8))
##ax1 = fig.add_subplot(211)
##fig = sm.graphics.tsa.plot_acf(series1,lags=40,ax=ax1)
#ax2 = fig.add_subplot(212)
#fig = sm.graphics.tsa.plot_pacf(series1,lags=40,ax=ax2)
##plt.show()
