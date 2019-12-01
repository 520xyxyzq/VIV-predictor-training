# Tuning parameters by runing the main program multiple times
from test import *

TRL = []; TEL=[];

for ii in range(1):    
    trl, tel = cross_val(N=10, layers=2,units=500, optim = 3, epochs = 100, plot = True)
    TRL += [trl]
    TEL += [tel]

print(TRL, TEL)
