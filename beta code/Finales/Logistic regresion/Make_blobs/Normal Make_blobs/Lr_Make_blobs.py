import numpy as np
import matplotlib.pyplot as plt
np.random.seed(6)
import math
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs
import pandas as pd
import timeit


N= 1000
IT= 1000
K= 20
text = 'Test'+str(IT)

(X,y) =  make_blobs(n_samples=5000,n_features=2,centers=2,cluster_std=1.05,random_state=20)
#we need to add 1 to X values (we can say its bias)
X1 = np.c_[np.ones((X.shape[0])),X]

plt.scatter(X1[:,1],X1[:,2],marker='o',c=y)
#plt.show()

plt.scatter(X1[:,1],y,marker='o',c=y)
#plt.show()

#random uniform distrubution weights
W=np.random.uniform(size=X1.shape[1])


def sigmoid(x):
    return float(1.0 / float((1.0 + np.exp(-1.0*x))))

sx=range(-10,10)
sy=[]
for i in sx:
    sy.append(sigmoid(i))

plt.plot(sx,sy)
#plt.show()

def predict():
    predicted_y=[]
    
    for x in X1:
        
        logit = x.dot(W) 
        predicted_y.append(sigmoid(logit)) 
        
    return np.array(predicted_y)

def cost_function(predicted_y):
    
    error=(-y*np.log(predicted_y)) - ((1-y)*np.log(1-predicted_y))
    cf=(1/X1.shape[0])*sum(error)
    
    return cf,error

def gradient_descent(lrate,epochs):
    
    total_expected_error=float("inf")
    errorlist=[]
    finalepoch=0
    
    for epoch in range(epochs):
        global W
        
        predictedY=predict() 
        total_error,error = cost_function(predictedY)
        gradient=X1.T.dot(error)/X1.shape[0]
        if epoch%10==0:
            errorlist.append(total_error)
            finalepoch+=1
          
        if (total_expected_error<total_error):
            return errorlist,finalepoch
            
        total_expected_error=total_error
        for (i,w) in enumerate(gradient):
            W[i]+=float(-lrate)*w
            

	return errorlist,finalepoch, W[i]
            
    
t=[]
l= []

for j in xrange(K):
	start = timeit.default_timer()
	total_error,finalepoch, cost=gradient_descent(0.001,10)
	l.append(cost)
	stop = timeit.default_timer()
	time = (stop-start)
	t.append(time)
#plotting 
plt.plot(range(finalepoch),total_error)
plt.xlabel("epochs in 10's")
plt.ylabel("error")
#plt.show()


yhat= predict() # we get the probablities scores (between 0 and 1)

#if the score is above 0.5 lets make it 1 else make it 0
for i,v in enumerate(yhat):
    if v >=0.56: 
        yhat[i]=1
    else:
        yhat[i]=0

print yhat.astype(int)

#actual y
print y


#error and acuracy 
error=sum((yhat-y)**2)
print(error)
accuracy=1-(error/100)
print accuracy 


plt.scatter(X1[:,1],X1[:,2],marker='o',c=yhat)
#plt.show()

writer = pd.ExcelWriter(text+'.xlsx', engine='xlsxwriter')
data_insertion={'Data':'Make_blobs','Algorithm':'Logistic Regression','Iterations':IT,'Loss':l,'Time':t}
df=pd.DataFrame(data_insertion)
df.to_excel(writer, sheet_name='Data')
df.mean().to_excel(writer, sheet_name='Mean')
df.var().to_excel(writer, sheet_name='Variance')
writer.save()



print"#Result:"
print " "
print df
print " "
print "Mean:"
print " "
print df.mean()
print " "
print "Variance:"
print " "
print df.var()

