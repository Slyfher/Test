from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = (x**2)*exp(x**5)
print('gradient with ad =',y.d(x))







