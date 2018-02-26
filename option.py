class Option : 
    def __init__(self, K, T):
        self.K = K
        self.T = T
    def payoff(self, underlying_path):
        pass
#Call Option class
class CallOption(Option):
    def payoff(self, underlying_path):
        return max(underlying_path[-1] - self.K, 0)
#Put Option class
class PutOption(Option):
    def payoff(self, underlying_path):
        return max(self.K - underlying_path[-1], 0)