import autograd.numpy as np
from autograd import grad,elementwise_grad
import pandas as pd
import timeit


text = []
r = []
t = []


def shafferN4(x, y): return 0.5+ ((np.cos(np.sin(np.absolute(x**2-y**2)))**2)-0.5/(1+0.001*(x**2+y**2))**2)


text.append('Gradient df/dx')
text.append('Gradient df/dy')
start = timeit.default_timer()
dfdx = grad(shafferN4,0)
r.append(dfdx(2.0,20.0))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
start = timeit.default_timer()
dfdy = grad(shafferN4,1)
r.append(dfdy(2.0,20.0))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
print " "
print "Sumary"
print " "
data_insertion={'A_Funtion':'shafferN4','B_Tool':'Autograd','D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)

print df