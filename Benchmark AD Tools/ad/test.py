from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = (5*x**2+7*x+2)**2/(x**2+6)
print('gradient with ad =',y.d(x))



