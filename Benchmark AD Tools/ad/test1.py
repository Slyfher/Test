from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = exp(sin(x))/(cos(x))
print('gradient with ad =',y.d(x))
