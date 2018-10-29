from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import autograd.numpy as np
import autograd.numpy.linalg as LA
from autograd import grad
import softmax
import time
import sys 

def numerical_grad(par):
    par={'weights':par[0],
        'bias':par[1]}
    hyper={'alpha':1e-2}
    smax_grad=softmax.grad(inputs,targets,par,hyper)
    w_grad=smax_grad['weights']
    b_grad=smax_grad['bias']
    return [w_grad,b_grad]


try: 
  n_data=np.int(sys.argv[1])
  dim=np.int(sys.argv[2])
  classes=np.int(sys.argv[3])
except:
  print("usage :", sys.argv[0]," n_data dim classes")
  sys.exit()

inputs = np.random.rand(n_data,dim)
targets = np.random.multinomial(1,[1/np.float(classes)]*classes,n_data)



# Check the gradients numerically, just to be safe.
weights = np.zeros((dim,classes))
bias = np.zeros(classes)
params=[weights,bias]
text = 'Test'+str(n_data)+str(dim)+str(classes)
timeGradNum = []

for k in range(20):

	# Optimize weights using gradient descent.
	t0=time.clock()
	n_grad=numerical_grad(params)
	t1=time.clock()
	timeGradNum.append(t1-t0)
	


writer = pd.ExcelWriter(text+'.xlsx', engine='xlsxwriter')
data_insertion={'U_Data':n_data,'Algorithm':'Softmax_regression','V_DIM':dim,'W_CLASSES':classes,\
'X_TimeGradNum':timeGradNum}

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

