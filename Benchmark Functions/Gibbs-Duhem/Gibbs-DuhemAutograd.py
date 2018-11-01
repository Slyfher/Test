import autograd.numpy as np
from autograd import grad, jacobian
import pandas as pd
import timeit

text = []
tool = []
r = []
t = []

tex= 'Gibbs-DuhemAutograd'

A12, A21 = 2.04, 1.5461  # Acetone/water https://en.wikipedia.org/wiki/Margules_activity_model
x, y = 1.0, 2.0

def GexRT(x, y):
    n = x + y
    x1 = x / n
    x2 = y / n
    return n * x1 * x2 * (A21 * x1 + A12 * x2)



tool.append('Autograd')
tool.append('Autograd')
tool.append('Analytical')
tool.append('Analytical')
text.append('Gradient df/dx')
text.append('Gradient df/dy')
text.append('Gradient df/dx')
text.append('Gradient df/dy')
start = timeit.default_timer()
lngamma1 = grad(GexRT)     # dGex/dx
r.append(lngamma1(x, y))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
start = timeit.default_timer()
lngamma2 = grad(GexRT,1)     # dGex/dy
r.append(lngamma2(x, y))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)

x, y = 1.0, 2.0
n = x + y
x1 = x / n
x2 = y / n

start = timeit.default_timer()
r.append((A12 + 2 * (A21 - A12) * x1) * x2**2)
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
start = timeit.default_timer()
r.append((A21 + 2 * (A12 - A21) * x2) * x1**2)
stop = timeit.default_timer()
result = (stop-start)
t.append(result)


print " "
print "Sumary"
print " "
writer = pd.ExcelWriter(tex+'.xlsx', engine='xlsxwriter')
data_insertion={'A_Funtion':'Gibbs-Duhem','B_Tool':tool,'D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Sumary')
writer.save()


print df
