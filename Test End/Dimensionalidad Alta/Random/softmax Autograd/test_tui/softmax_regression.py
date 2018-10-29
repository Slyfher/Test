from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import autograd.numpy as np
import autograd.numpy.linalg as LA
from autograd import grad
import softmax
import time
import sys 



def training_loss(par):
    par={'weights':par[0],
        'bias':par[1]}
    hyper={'alpha':1e-2}
    return -softmax.loss(inputs,targets,par,hyper)

# Build a toy dataset.
try: 
  n_data=np.int(sys.argv[1])
  dim=np.int(sys.argv[2])
  classes=np.int(sys.argv[3])
except:
  print("usage :", sys.argv[0]," n_data dim classes")
  sys.exit()

inputs = np.random.rand(n_data,dim)
targets = np.random.multinomial(1,[1/np.float(classes)]*classes,n_data)

# Build a function that returns gradients of training loss using autograd.
training_gradient_fun = grad(training_loss)

# Check the gradients numerically, just to be safe.
weights = np.zeros((dim,classes))
bias = np.zeros(classes)
params=[weights,bias]
text = 'Test'+str(n_data)+str(dim)+str(classes)
timeGraNum = []
timeAutograd= []
addL2Norm = []

for k in range(20):

	# Optimize weights using gradient descent.
	t2=time.clock()
	auto_grad=training_gradient_fun(params)
	t3=time.clock()
	timeAutograd.append(t3-t2)



writer = pd.ExcelWriter(text+'.xlsx', engine='xlsxwriter')
data_insertion={'U_Data':n_data,'Algorithm':'Softmax_regression','V_DIM':dim,'W_CLASSES':classes,\
'TimeAutograd':timeAutograd}

df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Data')
df.mean().to_excel(writer, sheet_name='Mean')
df.var().to_excel(writer, sheet_name='Variance')
writer.save()



print(" ")
print("Result:")
print(df)
print(" ")
print("Mean:")
print(df.mean())
print(" ")
print("Variance:")
print(df.var())

