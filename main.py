import numpy as np
import matplotlib.pyplot as plt
from black_sholes import BS
from option import CallOption
from model import HestonModel

#Number of time steps
nSteps = 50
#Number of monte carlo paths
nPaths = 10000
#Strike
#K = 145
#maturity
T = 1.1
#Initial stock price
s0 = 1162.0 #GOOGL
#Initial volatility
v0 = 0.3371
#risk free rate
r = 0.0127
#long term volatility(equiribrium level)
theta = 0
#Mean reversion speed of volatility
kappa = 46.7194
#lambda(volatility of Volatility)
lamda = 0.0491
#rho
rho = 0.059

#simulation
K = np.arange(1140, 1200, 10)
actual = [12.56, 9.55, 7.10, 5.40, 3.90, 2.80, 1.94, 1.41]
heston_ = []
print ("Strike:","Heston:    ", "Actual :   ","Difference:")
i = 0
for k in K:
    call_option = CallOption(k, T)
    heston = HestonModel(nSteps, nPaths, s0, v0, r, theta, kappa, lamda, rho)
    price = heston.price(call_option)
    heston_.append(price)
    print (k,"   ", price, actual[i],"",(price-actual[i])/actual[i])
    i = i+1
    


#plot result
plt.plot(K, heston_,"b")
plt.plot(K, actual,"r")
plt.xlabel('Strike (K)')
plt.ylabel('Option price')
plt.title('JPM')
plt.show()
