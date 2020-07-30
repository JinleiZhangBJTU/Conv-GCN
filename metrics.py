from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from math import sqrt
import numpy as np


def weighted_mean_absolute_percentage_error(Y_true, Y_pred):
	# (n * 276)
	total_sum=np.sum(Y_true)
	average=[]
	for i in range(len(Y_true)):
		for j in range(len(Y_true[0])):
			if Y_true[i][j]>0:
				temp=(Y_true[i][j]/total_sum)*np.abs((Y_true[i][j] - Y_pred[i][j]) / Y_true[i][j])
				average.append(temp)
	return np.sum(average)

def evaluate_performance(Y_test_original,predictions):
	RMSE = sqrt(mean_squared_error(Y_test_original, predictions))
	print('RMSE is '+str(RMSE))
	R2 = r2_score(Y_test_original,predictions)
	print("R2 is："+str(R2))
	MAE=mean_absolute_error(Y_test_original, predictions)
	print("MAE is："+str(MAE))
	WMAPE=weighted_mean_absolute_percentage_error(Y_test_original,predictions)
	print("WMAPE is "+str(WMAPE))
	return RMSE,R2,MAE,WMAPE