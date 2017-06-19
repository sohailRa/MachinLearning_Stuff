# Credits to Siraj

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt 

plt.switch_backend('TkAgg')

# Lists to store data from csv file
dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvFile:
		fileReader = csv.reader(csvFile)
		next(fileReader) # skip first row
		for row in fileReader:
			dates.append(int(row[0].split('-')[0]))
			prices.append(float(row[1]))
	return

def prdict_prices(dates, prices, x):
	dates = np.reshape(dates,(len(dates), 1)) # converting to n x 1 matrix

	svr_lin = SVR(kernel= 'linear', C= 1e3)
	#svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
	print("Training RBF Model")
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	print("Training Linear Model")
	svr_lin.fit(dates, prices)
	#print("Training Polynomial model")
	#svr_poly.fit(dates, prices)
	print("Plotting Graph")
	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	#plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
	plt.xlabel('Date')
	plt.ylabel('Price')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()

	#return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0]

get_data('aapl.csv')
print("done!")
predicted_prices = prdict_prices(dates, prices, 29)