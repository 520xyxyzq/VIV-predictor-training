import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

def cross_val(N=10,layers=2,units=500,optim=3,epochs=100,mode="cv",plot = False):

    Xtrain = genfromtxt('Xtrain.csv', delimiter=',')
    Ytrain = genfromtxt('Ytrain2.csv', delimiter=',')
    Xtest = genfromtxt('Xtest.csv', delimiter=',')
    Ytest = genfromtxt('Ytest2.csv', delimiter=',')
    xhigh = np.array([[2.1686,2.6595,-0.7179,6]]).T
    xmulti = np.array([[1.0318, 1.5211, -1.1854, 5]]).T
    
    # Parameters to tune:

    # N = 20;       # num of sine bases (odd order) approximating CL
    # layers = 2;   # num of hidden layers
    # units = 500   # num of units in each layer
    # optim = 1;    # optimizer 1 for sgd, 2 for adadelta, 3 for adam
    # epochs = 100; # num of epochs
    # plot = False  # plot or not in each iteration 

    L = 7.9; n = 98 + 1;
    x =np.array([np.linspace(0, L, n)]).T; mm = np.array([np.arange(1,2*N,2)]);
    sinbase = np.sin(np.pi*x.dot(mm)/L);

    
    train_loss, test_loss = [], []
    if mode == "cv":
        fold = 10
    else:
        fold  = 1
   
    for tt in range(fold):

        print("Fold",tt+1)
        
        x_train = Xtrain[:,tt*63:tt*63+63]
        x_test = Xtest[:,tt*7:tt*7+7]
        y_train = Ytrain[:,tt*63:tt*63+63]
        y_test = Ytest[:,tt*7:tt*7+7]

        model = Sequential()

        model.add(Dense(units=units, activation='relu', input_dim=x_train.shape[0]))
        for ll in range(layers - 1):
            model.add(Dense(units=units, activation='relu'))
        model.add(Dense(units=N, activation='linear'))

        if optim == 1:
            optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        elif optim == 2:
            optimizer = optimizers.Adadelta(lr=0.10)
        elif optim == 3:
            optimizer = optimizers.Adam()
            
        def test(y_true, y_pred):
            if not K.is_tensor(y_pred): # if y_pred is not a keras tensor
                y_pred = K.constant(y_pred) # change it to a constant tensor
                y_true = K.cast(y_true, y_pred.dtype) # cast dtype of y_true to the same dtype as y_pred
            return K.mean(K.square(y_true-y_pred), axis = -1)

        def myloss(y_true, y_pred):
            if not K.is_tensor(y_pred):
                y_pred = K.constant(y_pred)
                y_true = K.cast(y_true, y_pred.dtype)

            L = 7.9; n = 98 + 1;
            x =np.array([np.linspace(0, L, n)]).T; mm = np.array([np.arange(1,2*N,2)]);
            sinbase = np.sin(np.pi*x.dot(mm)/L);
        
            sinbase = K.constant(sinbase)
            sinbase = K.cast(sinbase, y_pred.dtype)
            y_pred = K.transpose(K.dot(sinbase, K.transpose(y_pred)))
            return K.mean(K.square(y_true-y_pred), axis = -1)

        model.compile(loss=myloss, optimizer=optimizer)

        hist = model.fit(x_train.T, y_train.T, epochs=epochs,
                 batch_size=63, validation_data=(x_test.T, y_test.T))

        if plot:
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.show()
    
        train_loss += [hist.history['loss'][-1]]
        test_loss += [hist.history['val_loss'][-1]]
    
        y_pred = model.predict(x_test.T).T
        if mode == "high":
            yhigh = model.predict(xhigh.T).T
        if mode == "multi":
            ymulti = model.predict(xmulti.T).T

        if plot:
            plt.plot(x, (y_test[:,0]))
            plt.plot(x, sinbase.dot(y_pred[:,0]));
            plt.show()
    
    if mode == "cv":    
        return np.mean(train_loss), np.mean(test_loss)
    elif mode == "high":
        plt.plot(x, sinbase.dot(yhigh))        
        plt.show()
        return (sinbase.dot(yhigh)).T
    elif mode == "multi":
        plt.plot(x, sinbase.dot(ymulti))        
        plt.show()
        return (sinbase.dot(ymulti)).T
    else:
        return hist.history['loss'],hist.history['val_loss']
