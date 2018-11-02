from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = (5*x**4-3*x**2+2*x)*exp(-3*x**2+x-2)
print('gradient with ad =',y.d(x))

