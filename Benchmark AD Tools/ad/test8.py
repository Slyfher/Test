from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = (x**2+3*x+6)**4/(x+1)
print('gradient with ad =',y.d(x))


