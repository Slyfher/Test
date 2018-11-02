from ad import adnumber
from ad.admath import * 

x = adnumber(2.0)
y = cos(exp(x))/x
print('gradient with ad =',y.d(x))









