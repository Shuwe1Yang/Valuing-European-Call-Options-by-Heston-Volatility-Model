#Class of Heston Stochastic Volatility Model
#Using Euler Discretization and Milstein Discretization
import numpy as np
from numpy.random import standard_normal

class HestonModel: 
    def __init__(self, nSteps, nPaths, s0, v0, r,theta, kappa, lamda, rho):
        self._nSteps = nSteps
        self._nPaths = nPaths
        #Initial stock price
        self._s0 = s0
        #Initial volatility
        self._v0 = v0
        #risk free rate
        self._r = r
        #long term volatility(equiribrium level)
        self._theta = theta
        #Mean reversion speed of volatility
        self._kappa = kappa
        #lambda(volatility of Volatility)
        self._lamda = lamda
        #rho
        self._rho = rho
    #Euler Discretization
    def _generate_path(self, dt):
        s = np.zeros(self._nSteps + 1)
        v = np.zeros(self._nSteps + 1)
        s[0] = self._s0            
        v[0] = self._v0
        dW1 = standard_normal(self._nSteps)
        dW2 = self._rho * dW1 + (1 - self._rho**2)**(0.5) * standard_normal(self._nSteps)
        for j in range(0, self._nSteps):
            s[j + 1] = s[j] * np.exp((self._r - 0.5 * v[j]) * dt + (v[j] * dt)**(0.5) * dW1[j])
            v[j + 1] = max(v[j] + (self._kappa * (self._theta - v[j]) * dt) + self._lamda * (v[j] * dt)**(0.5) * dW2[j], 0)
        return s
    #Milstein Discretization
    def _generate_path_Mil(self, dt):
        s = np.zeros(self._nSteps + 1)
        v = np.zeros(self._nSteps + 1)
        s[0] = self._s0            
        v[0] = self._v0
        dW1 = standard_normal(self._nSteps)
        dW2 = self._rho * dW1 + (1 - self._rho**2)**(0.5) * standard_normal(self._nSteps)
        for j in range(0, self._nSteps):
            s[j + 1] = s[j] * np.exp((self._r - 0.5 * v[j]) * dt + (v[j] * dt)**(0.5) * dW1[j])
            v[j + 1] = max((self._kappa*(self._theta - v[j])*dt)-(0.25*(self._lamda**2)*dt)+\
                       (v[j]**0.5+0.5*self._lamda*dW2[j]*(dt**(0.5)))**2, 0)
        return s            
    #Pricing with Euler Discretization
    def price(self, option):
        payOff_Sum = 0.0
        for i in range(0, self._nPaths):
            payOff_Sum += option.payoff(self._generate_path(option.T / self._nSteps))
        return (np.exp(- self._r * option.T) * payOff_Sum / self._nPaths)
    #Pricing with Milstein Discretization
    def price_Mil(self, option):
        payOff_Sum = 0.0
        for i in range(0, self._nPaths):
            payOff_Sum += option.payoff(self._generate_path_Mil(option.T / self._nSteps))
        return (np.exp(- self._r * option.T) * payOff_Sum / self._nPaths)
    