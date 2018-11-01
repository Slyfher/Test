import autograd.numpy as np
from autograd import grad, jacobian
import pandas as pd
import timeit
E0, B0, BP, V0 = -56.466,   0.49,    4.753,  16.573

text = []
tool = []
r = []
t = []
tex= 'MurnaghanAutograd'

tool.append('Autograd')
tool.append('Autograd')
text.append('(P(V0))')
text.append('P(0.99 * V0)')


def Murnaghan(vol):
    E = E0 + B0 * vol / BP * (((V0 / vol)**BP) / (BP - 1.0) + 1.0) - V0 * B0 / (BP - 1.)
    return E

def P(vol):
    dEdV = grad(Murnaghan)
    return -dEdV(vol) * 160.21773  # in Gpa


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
writer = pd.ExcelWriter(tex+'.xlsx', engine='xlsxwriter')
data_insertion={'A_Funtion':'Murnaghan','B_Tool':tool,'D_Diff':text,'E_Result':r,'F_Time':t}
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Sumary')
writer.save()

print df