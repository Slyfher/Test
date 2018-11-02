from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = (4*x**6+5*x+3)*(exp(-x**2+5*x+1))
print('gradient with ad =',y.d(x))





