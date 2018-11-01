import autograd.numpy as np
from autograd import grad,elementwise_grad
import pandas as pd
import timeit

text = []
r = []
t = []
tex= 'ackleyAutograd'

def ackley(x, y): return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+(np.cos(2*np.pi*y))))+np.exp(1)+20

text.append('Gradient df/dx')
text.append('Gradient df/dy')
start = timeit.default_timer()
dfdx = grad(ackley,0)
r.append(dfdx(1.0,-1.0))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
start = timeit.default_timer()
dfdy = grad(ackley,1)
r.append(dfdy(1.0,-1.0))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
print " "
print "Sumary"
print " "
writer = pd.ExcelWriter(tex+'.xlsx', engine='xlsxwriter')
data_insertion={'A_Funtion':'Ackley','B_Tool':'Autograd','D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Sumary')
writer.save()

print df

