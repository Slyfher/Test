import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import pystan
import datetime
import pandas as pd
import timeit
import xlsxwriter
import  logging
logging.getLogger("pystan").propagate=False
logging.getLogger("stan").propagate=False
logging.getLogger("numpy").propagate=False
logging.getLogger("sklearn").propagate=False

iterations=10000 #iterations of algorithm
K = 1 #times to run each algorithm

text = 'test_iter_'+str(iterations) #name prefix of output file

time_nuts = 0.0 #initial time for nuts
time_hmc = 0.0# initial time for hmc

nuts_time_array = []
hmc_time_array = []

#load dataset
diabetes = datasets.load_diabetes()
min_max_scaler = MinMaxScaler()#establece un minimo y max de valores de muestra
diabetes_X = diabetes.data
diabetes_X = min_max_scaler.fit_transform(diabetes_X)#ajuste de data
print "min max scaler values: ",min_max_scaler

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

print "#Horseshoe-prior with No-U-Turn-Sampler"
print "#Parameters:\n"
print "#Dataset: diabetes"
print "#N Itertations:", iterations

print "\n#Train:"
print "#Horseshoe-prior\n"

print "\n#TAKING TIME FOR NUTS..."
for k in range(K):
  start_nuts = timeit.default_timer()
  fit_nuts = pystan.stan(model_code=model_code, data=data, seed=5, iter=iterations, algorithm="NUTS")
  stop_nuts = timeit.default_timer()
  time_nuts = (stop_nuts-start_nuts)
  nuts_time_array.append(time_nuts)

print "\n#TAKING TIME FOR HMC..."
for k in range(K):
  start_hmc = timeit.default_timer()
  fit_hmc = pystan.stan(model_code=model_code, data=data, seed=5, iter=iterations, algorithm="HMC")
  stop_hmc = timeit.default_timer()
  time_hmc = (stop_hmc-start_hmc)
  hmc_time_array.append(time_hmc)


writer = pd.ExcelWriter(text+'.xlsx', engine='xlsxwriter')
data_insertion={'Iterations':iterations,'Time_nuts':nuts_time_array,'Time_hmc':hmc_time_array}
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Data')
df.mean().to_excel(writer, sheet_name='Mean')
df.var().to_excel(writer, sheet_name='Variance')
writer.save()

print "df:\n",df
print " "
print "Mean:\n",df.mean(),"\n"
print "Variance:\n",df.var(),"\n"
