import autograd.numpy as np
from autograd import grad,elementwise_grad
import pandas as pd
import timeit

text = []
r = []
t = []

tex= 'CrosstrayAutograd'

def Crosstray(x, y): return -0.0001*(np.absolute(np.sin(x)*np.sin(y)*np.exp(np.absolute(100-(np.sqrt(x**2+y**2)/np.pi))))+1)**0.1



text.append('Gradient df/dx')
text.append('Gradient df/dy')
start = timeit.default_timer()
dfdx = grad(Crosstray,0)
r.append(dfdx(1.0,-1.0))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
start = timeit.default_timer()
dfdy = grad(Crosstray,1)
r.append(dfdy(1.0,-1.0))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
print " "
print "Sumary"
print " "
writer = pd.ExcelWriter(tex+'.xlsx', engine='xlsxwriter')
data_insertion={'A_Funtion':'Cross in tray','B_Tool':'Autograd','D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Sumary')
writer.save()



print df

