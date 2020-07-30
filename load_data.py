from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import numpy as np
np.set_printoptions(threshold=np.inf)
import time
import csv

global_start_time = time.time()
def Get_All_Data(TG,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week):
	# deal with inflow data 处理进站数据
	metro_enter = []
	with open('data/inflowdata/in_'+str(TG)+'min.csv') as f:
		data = csv.reader(f, delimiter=",")
		for line in data:
			line=[int(x) for x in line]
			metro_enter.append(line)

	def get_train_data_enter(data,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week):
		data=np.array(data)
		data2=np.zeros((data.shape[0],data.shape[1]))
		a=np.max(data)
		b=np.min(data)
		for i in range(len(data)):
			for j in range(len(data[0])):
				data2[i,j]=round((data[i,j]-b)/(a-b),5)
		# 不包括第一周和最后一周的数据
		# not include the first week and the last week among the five weeks
		X_train_1 = [[] for i in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number)]
		Y_train = []
		for index in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number):
			for i in range(276):
				temp=data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i,index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i,index: index + time_lag-1])
				X_train_1[index-TG_in_one_week].append(temp)
			Y_train.append(data2[:,index + time_lag-1])
		X_train_1,Y_train = np.array(X_train_1), np.array(Y_train)
		print("X_train_1.shape,Y_train.shape")
		print(X_train_1.shape,Y_train.shape)

		X_test_1 = [[] for i in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1)]
		Y_test = []
		for index in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1):
			for i in range(276):
				temp=data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i,index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i,index: index + time_lag-1])
				X_test_1[index-(len(data2[0]) - TG_in_one_day*forecast_day_number)].append(temp)
			Y_test.append(data2[:,index + time_lag-1])
		X_test_1,Y_test = np.array(X_test_1), np.array(Y_test)
		print("X_test_1.shape,Y_test.shape")
		print(X_test_1.shape,Y_test.shape)

		Y_test_original = []
		for index in range(len(data[0]) - TG_in_one_day*forecast_day_number,len(data[0])-time_lag+1):
			Y_test_original.append(data[:,index + time_lag-1])
		Y_test_original = np.array(Y_test_original)
		print("Y_test_original.shape")
		print(Y_test_original.shape)

		return X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b

	#获取训练集和测试集，Y_test_original为没有scale之前的原始测试集，评估精度用，a,b分别为最大值和最小值
	#Get the training dataset and the test dataset, Y_test_original is the original test data before scaling, which can be used for evaluation.
	#a and b as the maximum and minimum values, respectively.
	X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b=get_train_data_enter(metro_enter,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week)
	print(a,b)

	#deal with outflow data. Similar with the inflow data while not including the testing data for outflow
	#处理出站数据
	metro_exit = []
	with open('data/outflowdata/out_'+str(TG)+'min.csv') as f:
		data = csv.reader(f, delimiter=",")
		for line in data:
			line=[int(x) for x in line]
			metro_exit.append(line)

	def get_train_data_exit(data,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week):
		data=np.array(data)
		data2=np.zeros((data.shape[0],data.shape[1]))
		a=np.max(data)
		b=np.min(data)
		for i in range(len(data)):
			for j in range(len(data[0])):
				data2[i,j]=round((data[i,j]-b)/(a-b),5)
		#不包括第一周和最后一周
		## not include the first week and the last week among the five weeks
		X_train_1 = [[] for i in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number)]
		for index in range(TG_in_one_week, len(data2[0]) - time_lag+1 - TG_in_one_day*forecast_day_number):
			for i in range(276):
				temp=data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()#上周同一个时间段的数据
				temp.extend(data2[i,index-TG_in_one_day: index + time_lag-1-TG_in_one_day])#前一天同一个时间段的数据
				temp.extend(data2[i,index: index + time_lag-1])#当天前几个时间段的数据
				X_train_1[index-TG_in_one_week].append(temp)
		X_train_1= np.array(X_train_1)
		print(X_train_1.shape)#其形状应该是(sample number, 276, 5, channel=3),3代表着上一周，前一天，当天，相当于275*5*3的图片

		X_test_1 = [[] for i in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1)]
		for index in range(len(data2[0]) - TG_in_one_day*forecast_day_number,len(data2[0])-time_lag+1):
			#此处注意test的下标要从0开始，而data2_all的下标要从
			for i in range(276):
				temp=data2[i,index-TG_in_one_week: index + time_lag-1-TG_in_one_week].tolist()
				temp.extend(data2[i,index-TG_in_one_day: index + time_lag-1-TG_in_one_day])
				temp.extend(data2[i,index: index + time_lag-1])
				X_test_1[index-(len(data2[0]) - TG_in_one_day*forecast_day_number)].append(temp)
		X_test_1= np.array(X_test_1)
		print(X_test_1.shape)
		return X_train_1,X_test_1

	X_train_2,X_test_2=get_train_data_exit(metro_exit,time_lag,TG_in_one_day,forecast_day_number,TG_in_one_week)


	return X_train_1,Y_train,X_test_1,Y_test,Y_test_original,a,b,X_train_2,X_test_2