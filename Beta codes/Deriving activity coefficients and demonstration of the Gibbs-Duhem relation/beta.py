from ad import adnumber
from ad.admath import * 


A12, A21 = 2.04, 1.5461  # Acetone/water https://en.wikipedia.org/wiki/Margules_activity_model

def GexRT(x, y):
    n = x + y
    x1 = x / n
    x2 = y / n
    return n * x1 * x2 * (A21 * x1 + A12 * x2)

x = adnumber(1.0)
y = adnumber(2.0)
z = GexRT(x,y)

print('gradient with ad =',z.d(x), z.d(y))


x, y = 1.0, 2.0
n = x + y
x1 = x / n
x2 = y / n

# Compare that to these analytically derived activity coefficients
print('Analytical:       ', (A12 + 2 * (A21 - A12) * x1) * x2**2, (A21 + 2 * (A12 - A21) * x2) * x1**2)



