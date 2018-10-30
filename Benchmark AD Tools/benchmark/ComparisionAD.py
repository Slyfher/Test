#benchmark Autograd vs AlgoPy
import autograd.numpy as NP
import pandas as pd
from autograd import grad,elementwise_grad
import numpy, algopy
from algopy import UTPM, exp, cos, sin, log
import timeit

#parameters
a = 2.0
F =[] #add function solve
SAP =[] #numerical solution AlgoPy
SAUT = [] #numerical solution Autograd
TAP =[] #time AlgoPy
TAUT =[] #time Autograd

###########################################################################################################################

def f0(x):   

	return (5*x**2+7*x+2)**2/(x**2+6)

F.append("(5*x**2+7*x+2)**2/(x**2+6)")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = f0(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx0= grad(f0)
SAUT.append(dfdx0(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)

###########################################################################################################################


def f1(x):   

	return(NP.exp(NP.sin(x)))/(NP.cos(x))

def fa1(x):   

	return (exp(sin(x)))/(cos(x)) 

F.append("(exp(sin(x)))/(cos(x))")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = fa1(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx1= grad(f1)
SAUT.append(dfdx1(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)

###########################################################################################################################

def f2(x):   

	return (7+2*x)**3/(x**3+4*x**2+1)

F.append("(7+2*x)**3/(x**3+4*x**2+1)")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = f2(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx2= grad(f2)
SAUT.append(dfdx2(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)

###########################################################################################################################
def f3(x):   

	return NP.log(1/((x**3-4*x+1)**2))

def fa3(x):   

	return log(1/((x**3-4*x+1)**2))

F.append("log(1/((x**3-4*x+1)**2))")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = fa3(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx3= grad(f3)
SAUT.append(dfdx3(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)

###########################################################################################################################
def f4(x):   

	return (NP.sin(NP.exp(x)))/(x)

def fa4(x):   

	return sin(exp(x))/(x)

F.append("sin(exp(x))/(x)")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = fa4(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx4= grad(f4)
SAUT.append(dfdx4(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
###########################################################################################################################

def f5(x): 

	return (x**3+4*x**2)*(5*x+4*x**2)**3

F.append("(x**3+4*x**2)*(5*x+4*x**2)**3")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = f5(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx5= grad(f5)
SAUT.append(dfdx5(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)

###########################################################################################################################

def f6(x):   
	
	return (4*x**3+3*x**2)*NP.exp(x**2+7)

def fa6(x):   
	
	return (4*x**3+3*x**2)*exp(x**2+7)

F.append("(4*x**3+3*x**2)*exp(x**2+7)")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = fa6(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx6= grad(f6)
SAUT.append(dfdx6(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)

###########################################################################################################################
def f8(x): 
	
	return (x**2+3*x+6)**4/(x+1)


F.append("(x**2+3*x+6)**4/(x+1)")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = f8(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx8= grad(f8)
SAUT.append(dfdx8(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)

###########################################################################################################################

def f9(x): 
	
	return (NP.exp(NP.sin(x)))/(NP.cos(x))

def fa9(x): 
	
	return (exp(sin(x)))/(cos(x))

F.append("(exp(sin(x)))/(cos(x))")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = fa9(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx9= grad(f9)
SAUT.append(dfdx9(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)


###########################################################################################################################

def f10(x): 
	
	return (4*x**6+5*x+3)*(NP.exp(-x**2+5*x+1))

def fa10(x): 
	
	return (4*x**6+5*x+3)*(exp(-x**2+5*x+1))

F.append("(4*x**6+5*x+3)*(exp(-x**2+5*x+1))")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = fa10(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx10= grad(f10)
SAUT.append(dfdx10(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)


###########################################################################################################################
def f11(x): 
	
	return (NP.cos(NP.exp(x)))/x

def fa11(x): 
	
	return (cos(exp(x)))/x

F.append("(cos(exp(x)))/x")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = fa11(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx11= grad(f11)
SAUT.append(dfdx11(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)


###########################################################################################################################
def f12(x): 
	
	return (x**2)*NP.exp(x**5)

def fa12(x): 
	
	return (x**2)*exp(x**5)

F.append("(x**2)*exp(x**5)")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = fa12(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx12= grad(f12)
SAUT.append(dfdx12(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)



###########################################################################################################################
def f13(x): 
	
	return (5*x**4-3*x**2+2*x)*NP.exp(-3*x**2+x-2)

def fa13(x): 
	
	return (5*x**4-3*x**2+2*x)*exp(-3*x**2+x-2)

F.append("(5*x**4-3*x**2+2*x)*exp(-3*x**2+x-2)")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = fa13(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx13= grad(f13)
SAUT.append(dfdx13(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)

###########################################################################################################################

def f14(x): 
	
	return ((6*x**2+x)**2)*((x**5+x**6)**4)

F.append("((6*x**2+x)**2)*((x**5+x**6)**4)")
start= timeit.default_timer()
cg = algopy.CGraph()
x = algopy.Function(a)
y = f14(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]
SAP.append(cg.gradient(a))
stop = timeit.default_timer()
result =(stop-start)
TAP.append(result)
start= timeit.default_timer()
dfdx14= grad(f14)
SAUT.append(dfdx14(a))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)

###########################################################################################################################

data_insertion={'A_Function':F,'D_AlgoPy solution':SAP,'E_Autograd solution':SAUT,'H_AlgoPy time':TAP,'I_Autograd time':TAUT}
#writer = pd.ExcelWriter(text+'.xlsx', engine='xlsxwriter')
df=pd.DataFrame(data_insertion)
#df.to_excel(writer, sheet_name='Sumary')
print "Sumary"
print ""
print df
#print "results save in:ComputeDerivativesSumary.xls"
#writer.save()


