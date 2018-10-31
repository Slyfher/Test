from ad import adnumber
from ad.admath import * 

def (x, y): return 0.5+ ((np.sin((x**2-y**2)))**2)-0.5/((1+0.001*(x**2+y**2))**2)


x = adnumber(2.0)
y = adnumber(20.0)
z = shafferN2(x,y)

print "gradient", z.d(x)
print "gradient", z.d(y)