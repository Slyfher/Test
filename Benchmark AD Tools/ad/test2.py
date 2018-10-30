from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = (7+2*x)**3/(x**3+4*x**2+1)
print('gradient with ad =',y.d(x))

