from ad import adnumber
from ad.admath import * 
import pandas as pd
import timeit

text = []
r = []
t = []

tex= 'CrosstrayAD'

def Crosstray(x, y): return -0.0001*(abs(sin(x)*sin(y)*exp(abs(100-(sqrt(x**2+y**2)/pi))))+1)**0.1


text.append('Gradient df/dx')
text.append('Gradient df/dy')
start = timeit.default_timer()
x = adnumber(1.0)
y = adnumber(-1.0)
z = Crosstray(x,y)
r.append(z.d(x))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
x = adnumber(1.0)
y = adnumber(-1.0)
z = Crosstray(x,y)
r.append(z.d(y))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)

print " "
print "Sumary"
print " "
writer = pd.ExcelWriter(tex+'.xlsx', engine='xlsxwriter')
data_insertion={'A_Funtion':'Cross in tray','B_Tool':'AD','D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Sumary')
writer.save()

print df


