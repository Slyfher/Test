from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = (4*x**3+3*x**2)*exp(x**2+7)
print('gradient with ad =',y.d(x))




