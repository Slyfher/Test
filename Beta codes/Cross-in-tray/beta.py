from ad import adnumber
from ad.admath import * 


def Crosstray(x, y): return -0.0001*(abs(sin(x)*sin(y)*exp(abs(100-(sqrt(x**2+y**2)/pi))))+1)**0.1

x = adnumber(1.0)
y = adnumber(-1.0)
z = Crosstray(x,y)

print "gradient", z.d(x)
print "gradient", z.d(y)

