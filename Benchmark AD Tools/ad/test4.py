from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = log(1/((x**3-4*x+1)**2))
print('gradient with ad =',y.d(x))


