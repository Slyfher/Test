from ad import adnumber
from ad.admath import *
import pandas as pd
import timeit


#parameters
text = 'BenchmarkAD'
F = [] #add function solve
SAD = [] #numerical solution AD
TAD = [] #time AD

###########################################################################################################################

F.append("(5*x**2+7*x+2)**2/(x**2+6)")
start= timeit.default_timer()
x = adnumber(2.0)
y = (5*x**2+7*x+2)**2/(x**2+6)
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("(exp(sin(x)))/(cos(x))")
start= timeit.default_timer()
x = adnumber(2.0)
y = exp(sin(x))/(cos(x))
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("(7+2*x)**3/(x**3+4*x**2+1)")
start= timeit.default_timer()
x = adnumber(2.0)
y = (7+2*x)**3/(x**3+4*x**2+1)
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("log(1/((x**3-4*x+1)**2))")
start= timeit.default_timer()
x = adnumber(2.0)
y = log(1/((x**3-4*x+1)**2))
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("sin(exp(x))/(x)")
start= timeit.default_timer()
x = adnumber(2.0)
y = sin(exp(x))/(x)
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("(x**3+4*x**2)*(5*x+4*x**2)**3")
start= timeit.default_timer()
x = adnumber(2.0)
y = (x**3+4*x**2)*(5*x+4*x**2)**3
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################
F.append("(4*x**3+3*x**2)*exp(x**2+7)")
start= timeit.default_timer()
x = adnumber(2.0)
y = (4*x**3+3*x**2)*exp(x**2+7)
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################


F.append("(x**2+3*x+6)**4/(x+1)")
start= timeit.default_timer()
x = adnumber(2.0)
y = (x**2+3*x+6)**4/(x+1)
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("(exp(sin(x)))/(cos(x))")
start= timeit.default_timer()
x = adnumber(2.0)
y = (exp(sin(x)))/(cos(x))
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("(4*x**6+5*x+3)*(exp(-x**2+5*x+1))")
start= timeit.default_timer()
x = adnumber(2.0)
y = (4*x**6+5*x+3)*(exp(-x**2+5*x+1))
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("(cos(exp(x)))/x")
start= timeit.default_timer()
x = adnumber(2.0)
y = cos(exp(x))/x
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("(x**2)*exp(x**5)")
start= timeit.default_timer()
x = adnumber(2.0)
y = (x**2)*exp(x**5)
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("(5*x**4-3*x**2+2*x)*exp(-3*x**2+x-2)")
start= timeit.default_timer()
x = adnumber(2.0)
y = (5*x**4-3*x**2+2*x)*exp(-3*x**2+x-2)
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

F.append("((6*x**2+x)**2)*((x**5+x**6)**4)")
start= timeit.default_timer()
x = adnumber(2.0)
y = ((6*x**2+x)**2)*((x**5+x**6)**4)
SAD.append(y.d(x))
stop = timeit.default_timer()
result =(stop-start)
TAD.append(result)

###########################################################################################################################

data_insertion={'A_Function':F,'G_AD solution':SAD,'L_AD time':TAD}
writer = pd.ExcelWriter(text+'.xlsx', engine='xlsxwriter')
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Sumary')
print "Sumary"
print ""
print df
print "results save in:ComputeDerivativesSumary.xls"
writer.save()

