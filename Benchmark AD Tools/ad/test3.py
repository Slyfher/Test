from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = (x**3+4*x**2)*(5*x+4*x**2)**3
print('gradient with ad =',y.d(x))


