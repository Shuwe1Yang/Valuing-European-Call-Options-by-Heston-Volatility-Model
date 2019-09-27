import numpy as np
from scipy.stats import norm
#Black Scholes Function
def BS(S, K, T, r, v, callPutFlag = 'c'):
    d1 = (np.log(S / K) + (r + 0.5 * v**2) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    if (callPutFlag == 'c') or (callPutFlag == 'C'):
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
#Calc Vega    
def BS_vega(S, K, T, r, v):
    d1 = (np.log(S/K)+(r+0.5*v**2)*T)/(v*np.sqrt(T))
    return (S*np.sqrt(T)*norm.cdf(d1))
#Calc Implied volatility using Newton's Method
def IV(price_, S, K, T, r, callPutFlag = 'c'):
    max_it = 200
    pre = 1.0e-5
    v = 0.2
    for i in range(max_it):
        price = BS(S, K, T, r, v, callPutFlag = 'c')
        vega = BS_vega(S, K, T, r, v)
        price = price 
        diff = price_ - price
        if (abs(diff)<pre):
            return v
        v = v + diff/vega
    #return best vol
    return v
# =============================================================================
# #test
# if __name__ == '__main__':
#     #correct : call 3.68
#print (BS(49.0, 50.0, 1.0, 0.01, 0.2, 'C'))
#     #correct : put 4.18
#     print (black_sholes(49.0, 50.0, 1.0, 0.01, 0.2, 'P'))
#     #correct : 0.2
#     print (implied_volatility(3.68, 49.0, 50.0, 1.0, 0.01, 'C'))
#     #correct : 0.2
#     print (implied_volatility(4.18, 49.0, 50.0, 1.0, 0.01, 'P'))
# =============================================================================
    
    

