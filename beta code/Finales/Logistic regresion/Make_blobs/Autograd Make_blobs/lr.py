import autograd.numpy as np
from autograd import grad
import pandas as pd
import timeit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets.samples_generator import make_blobs



N = 1000
IT = 1000
K = 20
text = 'Test'+str(N)+str(IT)

# Load a simple dataset

X, y = make_blobs(n_samples=N,n_features=2,centers=2,cluster_std=1.05,random_state=20)

# Normalize the input features
X = MinMaxScaler(feature_range=(-1,1)).fit_transform(X)


# Add a column of constant terms
X = np.hstack((X, np.ones((X.shape[0], 1))))

# Split the dataset in train and test part
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25)


# Sigmoid function
def Sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))




# Logistic regression model
def LogisticRegression(w, inp):
    pre = np.dot(inp, w)
    return Sigmoid(pre)

# Cross-entropy loss function
# Note the interaction between the logarithm and the exponential of LogisticRegression - there
# are better ways to implement this step.
def Cross_Entropy(w):
    pred = LogisticRegression(w, X_train)
    return - np.mean(y_train*np.log(pred) + (1.0 - y_train)*np.log(1.0 - pred))

 # Accuracy function
def Accuracy(y_true, y_pred):
    return np.mean(y_true == np.round(y_pred))



# Iteration steps
iters = IT

# Debug values
loss = np.zeros(iters)
acc = np.zeros(iters)
time = 0.0
alpha = 0.01
result = 0.0
n=[]
t=[]
l=[]
a=[]

# Initialize weights
w = np.random.randn(X_train.shape[1])*0.1

# Gradient function
grad_fcn = grad(Cross_Entropy)
print "#Logistic Regression with Autograd"
print " "
print "#Parameters:"
print " "
print "#Dataset: make_blobs"
print "#Training examples:",len(X_train)
print "#Features:", len(w)
print "#learning rate:", alpha
print "#N Itertations:", iters
print "N Repeat:", K



print "#Train:"
print " "
print "#Stochastic gradient descent:"
print " "



for k in range(K):
	start = timeit.default_timer()
	for i in range(iters):
	    loss[i] = Cross_Entropy(w)
	    acc[i] = Accuracy(y_train, LogisticRegression(w, X_train))
	    w = w - alpha*grad_fcn(w)
	stop = timeit.default_timer()
	time = (stop-start)

	n.append(k)
	l.append(loss[i])
	t.append(time)
	a.append(Accuracy(y_test, LogisticRegression(w, X_test)))




print " "
print "#Predict:"
print " "
print "#Parameters:"
print " "
print "#Test examples:",len(X_test)
print " "
writer = pd.ExcelWriter(text+'.xlsx', engine='xlsxwriter')
data_insertion={'Data':'Make_blobs','Algorithm':'Logistic Regression','Iterations':IT,'Loss':l,'Time':t,'XAccuracy':a}
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
