#Read The Data
#here we read the data and orgainze it in a file to be ready

__author__ = 'aqeel'
'''Train and evaluate a simple MLP on the Souq.com Reviews newswire topic classification task.
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python examples/NNClassifiyReviews.py
CPU run command:
    python examples/NNClassifiyReviews.py
'''
import numpy as np
np.random.seed(1337)
#random.seed(1337)
import keras
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Merge, Dropout, RepeatVector,MaxoutDense,Activation
from keras.layers import recurrent
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.objectives import mse as MMSE
from keras.optimizers import RMSprop
import theano
import theano.tensor as K
import math
import random

# #Read The Data

#mydata = pd.read_csv('All.csv',sep=',',quotechar='"')
#mydata = pd.read_csv('All_cleaned.csv',sep=',',quotechar='"')
mydata = pd.read_csv('All_programmers_outliered.csv',sep=',',quotechar='"')
#mydata = pd.read_csv('All_programmers.csv',sep=',',quotechar='"')
print len(mydata)
mydata =mydata.dropna()
print len(mydata)
MORE_THAN_MEAN_BY= 0
mydata['a_words'] = map(lambda x:x.split(),mydata['body'])
mydata['q_words'] = map(lambda x:x.split(),mydata['q_body'])
mydata['q_bdlen'] = map(lambda x:len(x),mydata['q_words'])
mydata['a_bdlen'] = map(lambda x:len(x),mydata['a_words'])
Q_MAX = max(mydata['q_bdlen'])
A_MAX = max(mydata['a_bdlen'])
Q_MEAN = int(math.floor(np.mean(mydata['q_bdlen']))+MORE_THAN_MEAN_BY)
A_MEAN=int(math.floor(np.mean(mydata['a_bdlen']))+MORE_THAN_MEAN_BY)
print min(mydata['q_bdlen']),Q_MAX,Q_MEAN-MORE_THAN_MEAN_BY,mydata['q_bdlen'].median()
print min(mydata['a_bdlen']),A_MAX,A_MEAN-MORE_THAN_MEAN_BY,mydata['a_bdlen'].median()
exit()
#Q_MEAN = 38 #median
#A_MEAN = 86 #median
def GetDataW(splitper=0.2):
    global mydata
    splitper = int(math.floor(splitper * len(mydata)) + 1)
    #Shuffle the data
    mydata = mydata.iloc[np.random.permutation(len(mydata))]
    mydata[:splitper].to_csv('test.csv')
    return mydata['q_words'][splitper:].tolist(),mydata['a_words'][splitper:].tolist(),    mydata['score'][splitper:].tolist(),mydata['q_words'][:splitper].tolist(),    mydata['a_words'][:splitper].tolist(),mydata['score'][:splitper].tolist()

print ('Loading Data....')
Qx_trn,Ax_trn,y_trn,Qx_test,Ax_test,y_test= GetDataW(0.2)
print ('Data Loaded')

All_Vocabulary={}
for i in Qx_trn+Ax_trn+Qx_test+Ax_test:
    for element in i:
        All_Vocabulary[element]=True

All_Vocabulary= All_Vocabulary.keys()
print len(All_Vocabulary)

vocab_size = len(All_Vocabulary)+1
word_idx = dict((c, i + 1) for i, c in enumerate(All_Vocabulary))

def Vectorize():
    global Qx_trn,Ax_trn,Qx_test,Ax_test,y_trn,y_test
    
    for i in range (0,len(Qx_trn)):
        Qx_trn[i] = [word_idx[l] for l in Qx_trn[i]]
    Qx_trn= pad_sequences(Qx_trn,Q_MEAN)
    
    for i in range (0,len(Ax_trn)):
        Ax_trn[i] = [word_idx[l] for l in Ax_trn[i]]
    Ax_trn =pad_sequences(Ax_trn,A_MEAN)
    
    for i in range (0,len(Qx_test)):
        Qx_test[i] = [word_idx[l] for l in Qx_test[i]]
    Qx_test= pad_sequences(Qx_test,Q_MEAN)
    
    for i in range (0,len(Ax_test)):
        Ax_test[i] = [word_idx[l] for l in Ax_test[i]]
    Ax_test = pad_sequences(Ax_test,A_MEAN)
    y_trn = np.array(y_trn)
    y_test = np.array(y_test)

Vectorize()

print('All Vocabulary  = {}'.format(vocab_size))
print('Train Questions.shape = {},Test Questions.shape{}'.format(Qx_trn.shape,Qx_test.shape))
print('Train Answers.shape = {},Test Answers.shape{}'.format(Ax_trn.shape,Ax_test.shape))
print('Train_Y.shape = {},Test_Y.shape{}'.format(y_trn.shape,y_test.shape))
print('Questions Max Length, Answers Max Length = {}, {}'.format(Q_MAX, A_MAX))


RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 4
LSTM_HIDDEN_SIZE = 4
D1Size = 100
D2Size = 100
#1-inf
BATCH_SIZE = 32
#1-inf
EPOCHS =5
#Done
#SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
theoptimizer = 'Adagrad'
#DONE
#0.1-0.9
thedropout =0.3
#DONE
#softmax,softplus,relu,tanh,sigmoid,hard_sigmoid,linear,
FirstActivation = 'relu'
SecondActivation='softmax'
#DONE
#mean_squared_error / mse,root_mean_squared_error / rmse,mean_absolute_error / mae,mean_absolute_percentage_error / mape
#mean_squared_logarithmic_error / msle,squared_hinge, hinge,binary_crossentropy: Also known as logloss,categorical_crossentropy: Also known as multiclass logloss. Note: using this objective requires that your labels are binary arrays of shape (nb_samples, nb_classes).
#poisson: mean of (predictions - targets * log(predictions))# cosine_proximity: the opposite (negative) of the mean cosine proximity between predictions and targets.
theloss='mse'
#DONE
#======================
print (theoptimizer)
print('Build model...')


Questions = Sequential()
Questions.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=Q_MEAN, mask_zero=True))
Questions.add(Dropout(thedropout))
Questions.add(RNN(LSTM_HIDDEN_SIZE, return_sequences=False))

Answers = Sequential()
Answers.add(Embedding(vocab_size, EMBED_HIDDEN_SIZE, input_length=A_MEAN,mask_zero=True))
Answers.add(Dropout(thedropout))
Answers.add(RNN(LSTM_HIDDEN_SIZE, return_sequences=False))

print Questions.output_shape
print Answers.output_shape
m = Merge([Questions, Answers], mode='concat')
print m.output_shape

def NMSE(y_true, y_pred):
    nmse = K.mean(K.square(y_pred-y_true)/(K.mean(y_pred)*K.mean(y_true)))
    results.write('\n MSE IS:= {}'.format(MMSE(y_true,y_pred)))
    return nmse

model = Sequential()
model.add(Merge([Questions, Answers], mode='concat'))

#model.add(RNN(EMBED_HIDDEN_SIZE, return_sequences=False))
model.add(Dropout(thedropout))
model.add(Dense(D1Size, activation='linear'))
model.add(Dense(D2Size, activation='linear'))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=theoptimizer,
              loss=theloss)

print('Training')
#kcb = keras.callbacks.ModelCheckpoint('tmp/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
hist = model.fit([Qx_trn, Ax_trn], y_trn, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05)#,callbacks =[kcb])
losst= model.evaluate([Qx_trn, Ax_trn], y_trn, batch_size=BATCH_SIZE)
#results.write('\n Train loss= {}'.format(losst))
loss= model.evaluate([Qx_test, Ax_test], y_test, batch_size=BATCH_SIZE)
results = open('results.txt','a')
results.write('\nBatchSize:{},EPOCH:{},Optimizer:{},Dropout:{},1stActivation:{},2ndActivation:{},theloss:{},QVectorLength:{},AVectorLength:{},D1Size:{},D2Size:{},LSTM_HS:{},EMB_HS:{}\n Train loss:{} \n Test loss:{}'\
.format(BATCH_SIZE,EPOCHS,theoptimizer,thedropout,FirstActivation,SecondActivation,theloss,Q_MEAN,A_MEAN,D1Size,D2Size,LSTM_HIDDEN_SIZE,EMBED_HIDDEN_SIZE,losst,loss))

results.close()
prediction_Y= model.predict([Qx_test, Ax_test],batch_size=BATCH_SIZE)
with open('RNNOUTPUT.csv','w') as output:
    output.write('\"id\",\"relevance\"\n')
    for i in range(len(prediction_Y)):
        output.write('{}'.format(prediction_Y[i][0])+'\n')


