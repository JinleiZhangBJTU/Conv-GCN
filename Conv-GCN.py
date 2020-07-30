from __future__ import print_function
from load_data import Get_All_Data
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import keras
from keras.layers import Input, Reshape, Conv3D, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from metrics import evaluate_performance
from keras.regularizers import l2
import numpy as np
from modifyGCNlayer import GraphConvolution1
from keras.utils import plot_model
import time
begintime = time.time()

X_train_1, Y_train, X_test_1, Y_test, Y_test_original, a, b, X_train_2, X_test_2 = \
    Get_All_Data(TG=30,time_lag=11,TG_in_one_day=36,forecast_day_number=5,TG_in_one_week=180)

adjacency = []
with open('data/adjacency.csv') as f:
    data = csv.reader(f, delimiter=",")
    for line in data:
        line=[float(x) for x in line]
        adjacency.append(line)
adjacency = np.array(adjacency)


# X_train_1,X_test_1 enter data

input1 = Input(shape=(X_train_1.shape[1],X_train_1.shape[2]))
out1 = GraphConvolution1(15, adj=adjacency, activation='relu', kernel_regularizer=l2(5e-4))(input1)
out1 = Reshape((276, 5, 3, 1), input_shape=(276, 15))(out1)

# X_train_2,X_test_2 exit data
input2 = Input(shape=(X_train_2.shape[1],X_train_2.shape[2]), name='input2')
out2 = GraphConvolution1(15, adj=adjacency, activation='relu', kernel_regularizer=l2(5e-4))(input2)
out2 = Reshape((276, 5, 3, 1), input_shape=(276, 15))(out2)

out = keras.layers.concatenate([out1, out2], axis=4)

out = Conv3D(16, kernel_size=3, padding='same', activation='relu')(out)

out = Flatten()(out)
out = Dense(276)(out)

model = Model(inputs=[input1, input2], outputs=[out])


# plot_model(model, to_file='model.png',show_shapes=True)
model.compile(loss='mse', optimizer=Adam(lr=0.001))#optimizer=RAdam(lr=0.001)
print("finish compile")


# # callbacks
# logdir = ".\callbacks"
# if not os.path.exists(logdir):
# 	os.mkdir(logdir)
# modelcheckpoint = os.path.join(logdir, "modelcheckpoint_best.h5")
# callbacks = [
# 	tf.keras.callbacks.TensorBoard(logdir),
# 	tf.keras.callbacks.ModelCheckpoint(modelcheckpoint,
# 									   save_best_only = True,
# 									   save_weights_only=True),
# 	tf.keras.callbacks.EarlyStopping(monitor="loss",
# 									 min_delta=1e-6,
# 									 patience=10)
# ]


def fit10():
    # model.load_weights('my_model_weights.h5')
    model.fit([X_train_1,X_train_2], Y_train, batch_size=64, epochs=10, verbose=1) # callbacks=callbacks,validation_split=0.2)
    Y_test_pre = model.predict([X_test_1,X_test_2], verbose=1)
    Y_test_pre = Y_test_pre.reshape(-1,276)
    Y_test_pre = Y_test_pre * a   # a = 8940 when TG = 30 min，a = 4744 when TG = 15 min
    model.save_weights('my_model_weights.h5')
    return evaluate_performance(Y_test_original,Y_test_pre)#


model.fit([X_train_1,X_train_2], Y_train, batch_size=64, epochs=10, verbose=1) # callbacks=callbacks,validation_split=0.2)
model.save_weights('my_model_weights.h5')
RMSEs = []
R2s = []
MAEs = []
WMAPEs = []
epochNum = 5
for i in range(epochNum):
    print(i)
    RMSE, R2, MAE, WMAPE = fit10()
    RMSEs.append(RMSE)
    R2s.append(R2)
    MAEs.append(MAE)
    WMAPEs.append(WMAPE)
plt.plot(RMSEs)

print("\n")
minindex=RMSEs.index(min(RMSEs))
print("RMSEs.index(min(RMSEs))", minindex)
print("min(RMSEs))", RMSEs[minindex])
print("min(MAEs))", MAEs[minindex])
print("min(WMAPEs))", WMAPEs[minindex])

print("\n")
minindex=MAEs.index(min(MAEs))
print("MAEs.index(min(MAEs))", minindex)
print("min(RMSEs))", RMSEs[minindex])
print("min(MAEs))", MAEs[minindex])
print("min(WMAPEs))", WMAPEs[minindex])

print("\n")
minindex=WMAPEs.index(min(WMAPEs))
print("WMAPEs.index(min(WMAPEs))", minindex)
print("min(RMSEs))", RMSEs[minindex])
print("min(MAEs))", MAEs[minindex])
print("min(WMAPEs))", WMAPEs[minindex])

plt.savefig('books_read.png')
plt.show()
totaltime=time.time()-begintime
print(totaltime)