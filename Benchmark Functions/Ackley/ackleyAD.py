from ad import adnumber
from ad.admath import * 
import pandas as pd
import timeit

text = []
r = []
t = []

tex= 'ackleyAD'

def ackley(x, y): return -20*exp(-0.2*sqrt(0.5*(x**2+y**2)))-exp(0.5*(cos(2*pi*x)+(cos(2*pi*y))))+exp(1)+20


text.append('Gradient df/dx')
text.append('Gradient df/dy')
start = timeit.default_timer()
x = adnumber(1.0)
y = adnumber(-1.0)
z = ackley(x,y)
r.append(z.d(x))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
x = adnumber(1.0)
y = adnumber(-1.0)
z = ackley(x,y)
r.append(z.d(y))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)

print " "
print "Sumary"
print " "
writer = pd.ExcelWriter(tex+'.xlsx', engine='xlsxwriter')
data_insertion={'A_Funtion':'Ackley','B_Tool':'AD','D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Sumary')
writer.save()


print df
