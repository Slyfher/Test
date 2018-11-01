from ad import adnumber
from ad.admath import * 
import pandas as pd
import timeit

text = []
tool = []
r = []
t = []


A12, A21 = 2.04, 1.5461  # Acetone/water https://en.wikipedia.org/wiki/Margules_activity_model
x, y = 1.0, 2.0

def GexRT(x, y):
    n = x + y
    x1 = x / n
    x2 = y / n
    return n * x1 * x2 * (A21 * x1 + A12 * x2)


tool.append('AD')
tool.append('AD')
tool.append('Analytical')
tool.append('Analytical')
text.append('Gradient df/dx')
text.append('Gradient df/dy')
text.append('Gradient df/dx')
text.append('Gradient df/dy')

x = adnumber(1.0)
y = adnumber(2.0)
z = GexRT(x,y)
start = timeit.default_timer()
x = adnumber(1.0)
y = adnumber(2.0)
z = GexRT(x,y)
r.append(z.d(x))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
start = timeit.default_timer()
x = adnumber(1.0)
y = adnumber(2.0)
z = GexRT(x,y)
r.append(z.d(y))
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
data_insertion={'A_Funtion':'Gibbs-Duhem','B_Tool':tool,'D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)

print df
