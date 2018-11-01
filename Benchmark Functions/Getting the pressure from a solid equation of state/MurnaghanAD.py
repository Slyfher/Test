from ad import adnumber
from ad.admath import * 
import pandas as pd
import timeit
E0, B0, BP, V0 = -56.466,   0.49,    4.753,  16.573

text = []
tool = []
r = []
t = []

tool.append('AD')
tool.append('AD')
text.append('(P(V0))')
text.append('P(0.99 * V0)')

def Murnaghan(vol):
    E = E0 + B0 * vol / BP * (((V0 / vol)**BP) / (BP - 1.0) + 1.0) - V0 * B0 / (BP - 1.)
    return E

def P(vol):
    vol = adnumber(vol)
    z = Murnaghan(vol)
    dEdV = z.d(vol)
    return -dEdV * 160.21773  # in Gpa



start = timeit.default_timer()
r.append(P(V0))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)
start = timeit.default_timer()
r.append(P(0.99 * V0))
stop = timeit.default_timer()
result = (stop-start)
t.append(result)

print " "
print "Sumary"
print " "
data_insertion={'A_Funtion':'Murnaghan','B_Tool':tool,'D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)

print df