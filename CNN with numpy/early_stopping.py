from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.metrics import mean_squared_error as mse
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor as SGD

def estop(X, y):
	p20 = pf(degree = 20, include_bias = False)
	ss = StandardScaler()
	X = ss.fit_transform(X)
	X = p20.fit_transform(X)
	print(r)
	X1, y1, X_valid, y_valid = X[:100], y[:100], X[100:], y[100:]
	besterr = float('inf')
	sgd = SGD(eta0 = 0.002, random_state = 42)
	epoch = 500
	
	for i in range(epoch):
		sgd.partial_fit(X1, y1.ravel())
		pred = sgd.predict(X_valid)
		error = mse( y_valid, pred, squared = False)
		if error < besterr:
			besterr = error
			model = deepcopy(sgd)
			print(1)