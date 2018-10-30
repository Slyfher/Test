import autograd.numpy as NP
from autograd import grad,elementwise_grad
from scipy.misc import derivative
from sympy import pprint, Symbol
import pandas as pd
from sympy.functions import cos
from sympy.functions import sin
from sympy.functions import exp
from sympy.functions import log
from sympy.functions import tan
import timeit

#Parameters
text = 'ComputeDerivativesSumary'
x= 2.0
X = Symbol('X')
F =[] #add function solve
SA =[] #numerical solution analitical
SFF =[] #numerical solution finite diferences
SAUT = [] #numerical solution Autograd
SSYM = [] #Symbolic
TA =[] #time analitical
TFF =[] #time finite diferences
TAUT =[] #time Autograd
TSYM =[] #time symbolic

def f0(x):   

	return (5*x**2+7*x+2)**2/(x**2+6)

d0 = (5*X**2+7*X+2)**2/(X**2+6)
def analitical0(x): 

	return (2*(5*x**2+7*x+2)*(5*x**3+58*x+42))/(x**2+6)**2

dfdx0= grad(f0)
F.append("(5*x**2+7*x+2)**2/(x**2+6)")
start= timeit.default_timer()
SA.append(analitical0(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx0(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f0,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d0.diff(X))
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f1(x):   
	
	return (NP.exp(NP.sin(x)))/(NP.cos(x))

d1 = (exp(sin(X)))/(cos(X))
def analitical1(x): 

	return (NP.exp(NP.sin(x)))*(1+NP.tan(x)*(1/NP.cos(x)))

dfdx1= grad(f1)
F.append("NP.exp(NP.sin(x)))/(NP.cos(x)")
start= timeit.default_timer()
SA.append(analitical1(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx1(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f1,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d1.diff(X))
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f2(x):   

	return (7+2*x)**3/(x**3+4*x**2+1)

d2 = (7+2*X)**3/(X**3+4*X**2+1)

def analitical2(x): 

	return ((7+2*x)**2)*(-13*x**2-56*x+6)/(x**3+4*x**2+1)**2

dfdx2= grad(f2)
F.append("(7+2*x)**3/(x**3+4*x**2+1)")
start= timeit.default_timer()
SA.append(analitical2(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx2(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f2,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d2.diff(X))
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f3(x):   

	return NP.log(1/((x**3-4*x+1)**2))

d3 = log(1/((X**3-4*X+1)**2))

def analitical3(x): 

	return -((2*(3*x**2-4))/(x**3-4*x+1))

dfdx3= grad(f3)
F.append("NP.log(1/((x**3-4*x+1)**2)")
start= timeit.default_timer()
SA.append(analitical3(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx3(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f3,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d3.diff(X))
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f4(x):   

	return (NP.sin(NP.exp(x)))/(x)


d4 = (sin(exp(X)))/(X)

def analitical4(x): 

	return (NP.exp(x)*x*NP.cos(NP.exp(x))-NP.sin(NP.exp(x)))/(x**2)

dfdx4= grad(f4)
F.append("(NP.sin(NP.exp(x)))/(x)")
start= timeit.default_timer()
SA.append(analitical4(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx4(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f4,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d4.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f5(x): 

	return (x**3+4*x**2)*(5*x+4*x**2)**3

d5 = (X**3+4*X**2)*(5*X+4*X**2)**3

def analitical5(x): 

	return (x**4)*((5+4*x)**2)*(36*x**2+158*x+100)

dfdx5= grad(f5)
F.append("(x**3+4*x**2)*(5*x+4*x**2)**3")
start= timeit.default_timer()
SA.append(analitical5(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx5(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f5,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d5.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f6(x):   
	
	return (4*x**3+3*x**2)*NP.exp(x**2+7)

d6 = (4*X**3+3*X**2)*exp(X**2+7)

def analitical6(x): 

	return (2*x*NP.exp(x**2+7))*(4*x**3+3*x**2+6*x+3)

dfdx6= grad(f6)
F.append("(4*x**3+3*x**2)*NP.exp(x**2+7)")
start= timeit.default_timer()
SA.append(analitical6(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx6(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f6,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d6.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f7(x):   
	
	return NP.tan((4*x**4-2*x**2+(7/2)*x**-3+5)**-2)

d7 = tan((4*X**4-2*X**2+(7/2)*X**-3+5)**-2)

def analitical7(x): 

	return (-2*(16*x**3-4*x-(21/2)*x**-4))/((4*x**4-2*x**2+(7/2)*x**-3+5)**3)*NP.cos((4*x**4-2*x**2+(7/2)*x**-3+5)**3)**-2

dfdx7= grad(f7)
F.append("NP.tan((4*x**4-2*x**2+(7/2)*x**-3+5)**-2)")
start= timeit.default_timer()
SA.append(analitical7(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx7(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f7,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d7.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################
def f8(x): 
	
	return (x**2+3*x+6)**4/(x+1)


d8 = (X**2+3*X+6)**4/(X+1)

def analitical8(x): 

	return (x**2+3*x+6)**3*(7*x**2+17*x+6)/(x+1)**2

dfdx8= grad(f8)
F.append("(x**2+3*x+6)**4/(x+1)")
start= timeit.default_timer()
SA.append(analitical8(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx8(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f8,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d8.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)




#################################################################################################################################

def f9(x): 
	
	return (NP.exp(NP.sin(x)))/(NP.cos(x))

d9 = (exp(sin(X)))/(cos(X))

def analitical9(x): 

	return (NP.exp(NP.sin(x)))*(1+(NP.tan(x))*(1/NP.cos(x)))

dfdx9= grad(f9)
F.append("(NP.exp(NP.sin(x)))/(NP.cos(x))")
start= timeit.default_timer()
SA.append(analitical9(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx9(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f9,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d9.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f10(x): 
	
	return (4*x**6+5*x+3)*(NP.exp(-x**2+5*x+1))


d10 = (4*X**6+5*X+3)*exp(-X**2+5*X+1)

def analitical10(x): 

	return (-8*x**7+20*x**6+24*x**5-10*x**2+19*x+20)*NP.exp(-x**2+5*x+1)

dfdx10= grad(f10)
F.append("(4*x**6+5*x+3)*(NP.exp(-x**2+5*x+1))")
start= timeit.default_timer()
SA.append(analitical10(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx10(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f10,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d10.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f11(x): 
	
	return (NP.cos(NP.exp(x)))/x

d11 = (cos(exp(X)))/X

def analitical11(x): 

	return -(((x*NP.exp(x))*(NP.sin(NP.exp(x)))+NP.cos(NP.exp(x)))/(x**2))

dfdx11= grad(f11)
F.append("(NP.cos(NP.exp(x)))/x")
start= timeit.default_timer()
SA.append(analitical11(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx11(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f11,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d11.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f12(x): 
	
	return (x**2)*NP.exp(x**5)

d12 = (X**2)*exp(X**5)

def analitical12(x): 

	return (NP.exp(x**5))*x*(5*x**5+2)

dfdx12= grad(f12)
F.append("(x**2)*NP.exp(x**5)")
start= timeit.default_timer()
SA.append(analitical12(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx12(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f12,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d12.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f13(x): 
	
	return (5*x**4-3*x**2+2*x)*NP.exp(-3*x**2+x-2)

d13 = (5*X**4-3*X**2+2*X)*exp(-3*X**2+X-2)

def analitical13(x): 

	return (-30*x**5+5*x**4+38*x**3-15*x**2-4*x+2)*NP.exp(-3*x**2+x-2)

dfdx13= grad(f13)
F.append("(5*x**4-3*x**2+2*x)*NP.exp(-3*x**2+x-2)")
start= timeit.default_timer()
SA.append(analitical13(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx13(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f13,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d13.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

def f14(x): 
	
	return ((6*x**2+x)**2)*((x**5+x**6)**4)

d14 = ((6*X**2+X)**2)*((X**5+X**6)**4)


def analitical14(x): 

	return (2*((6*x**2+x)*((x**5+x**6)**3))*(84*x**7+85*x**6+11*x**5))

dfdx14= grad(f14)
F.append("((6*x**2+x)**2)*((x**5+x**6)**4)")
start= timeit.default_timer()
SA.append(analitical14(x))
stop = timeit.default_timer()
result =(stop-start)
TA.append(result)
start= timeit.default_timer()
SAUT.append(dfdx14(x))
stop = timeit.default_timer()
result =(stop-start)
TAUT.append(result)
start= timeit.default_timer()
SFF.append(derivative(f14,2.0,dx=1e-6))
stop = timeit.default_timer()
result =(stop-start)
TFF.append(result)
start= timeit.default_timer()
print ""
print "Symbolic solution:"
print ""
print ""
pprint(d14.diff(X))
print ""
SSYM.append("Symbolic")
stop = timeit.default_timer()
result =(stop-start)
TSYM.append(result)

#################################################################################################################################

data_insertion={'A_Function':F,'B_Analitical solution':SA,'C_Finite differences solution':SFF, 'D_Autograd solution':SAUT,\
'E_Symbolic solution':SSYM,'F_Analitical time':TA,'G_Finite differences time':TFF,'H_Autograd time':TAUT,'I_Symbolic time':TSYM}
writer = pd.ExcelWriter(text+'.xlsx', engine='xlsxwriter')
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Sumary')
print ""
print ""
print "results save in:ComputeDerivativesSumary.xls"
writer.save()