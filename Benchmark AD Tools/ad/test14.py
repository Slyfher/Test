from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = ((6*x**2+x)**2)*((x**5+x**6)**4)
print('gradient with ad =',y.d(x))

