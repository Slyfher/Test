from ad import adnumber
from ad.admath import * 
import pandas as pd
import timeit

text = []
r = []
t = []


def shafferN2(x, y): return 0.5+ ((sin((x**2-y**2)))**2)-0.5/((1+0.001*(x**2+y**2))**2)

text.append('Gradient df/dx')
text.append('Gradient df/dy')
start = timeit.default_timer()
x = adnumber(2.0)
y = adnumber(20.0)
z = shafferN2(x,y)
r.append(z.d(x))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
x = adnumber(2.0)
y = adnumber(20.0)
z = shafferN2(x,y)
r.append(z.d(y))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)

print " "
print "Sumary"
print " "
data_insertion={'A_Funtion':'shafferN2','B_Tool':'AD','D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)

print df
