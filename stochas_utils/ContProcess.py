import numpy as np
import matplotlib.pyplot as plt

class ContProcess():
    """
        Continuous Stochastic Process
    """
    def __init__(self):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError


class GenericCIRProcess(ContProcess):
    '''
        Generic Cox–Ingersoll–Ross Stochastic Process:
        dSt = k*(S_bar - St)^eta*dt + sigma*St^gamma*dWt
        params
        :k (float) mean-reverting speed
        :S_bar (float) long-term mean
        :S0 (float) initial value
        :eta (float)
        :sigma (float) volatility
        :gamma (float)
    '''
    def __init__(self, S0, k, S_bar, sigma, eta=1, gamma=.5):
        self.S0 = S0
        self.k = k 
        self.S_bar = S_bar 
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma

    def generate(self, T, n, N, method='euler'):
        """
            generate process on [0, T]
            params
            :n (int) number of discretization on [0, T]
            :N (int) number of paths
            :method (str) 
                euler: Euler Scheme
                doss: Euler-Doss Scheme
                milstein: Milstein Scheme
        """
        paths = []
        for _ in range(N):
            dWt = np.random.normal(0, np.sqrt(T/n), n)
            for j in range(n):
                if method == 'euler':
                    if j == 0:
                        path = [self.S0]
                    St = path[j]
                    St_plus = St + self.k*np.power(self.S_bar-St, eta)*T/n + self.sigma*np.power(St, self.gamma)*dWt[j]
                    path.append(St_plus)
                elif method == 'doss':
                    if j == 0:
                        Y0 = np.power(self.S0, 1-self.gamma)/(self.sigma*(1-self.gamma))
                        path = [Y0]
                    Yt = path[j]
                    Gt = np.power(self.sigma*(1-self.gamma)*Yt, 1/(1-self.gamma))
                    At = self.k * np.power(self.S_bar - Gt, eta)
                    Bt = self.sigma * np.power(Gt, self.gamma)
                    Bpt = self.sigma * self.gamma * np.power(Gt, self.gamma-1)
                    Yt_plus = Yt + (At / Bt - 0.5*Bpt)*T/n + dWt[j]
                    path.append(Yt_plus)
                    if j == n-1:
                        path = list(map(lambda x: np.power(self.sigma*(1-self.gamma)*x, 1/(1-self.gamma)), path))
                elif method == 'milstein':
                    if j == 0:
                        path = [self.S0]
                    St = path[j]
                    at = self.k * np.power(self.S_bar-St, self.eta)
                    bt = self.sigma * np.power(St, self.gamma)
                    bpt = self.sigma * self.gamma * np.power(St, self.gamma-1)
                    St_plus = St + at*T/n + bt*dWt[j] + 0.5*bt*bpt*(dWt[j]**2-T/n)
                    path.append(St_plus)
            paths.append(path)
        return paths

class CEVProcess(ContProcess):
    '''
        constant elasticity of variance Stochastic Process:
        dSt = a*St*dt + b*St^gamma*dWt
        params
        :S0 (float) initial value
        :a (float)
        :b (float)
        :gamma (float)
    '''
    def __init__(self, S0, a, b, gamma):
        self.S0 = S0
        self.a = a
        self.b = b
        self.gamma = gamma
    
    def generate(self, T, n, N, method='euler'):
        """
            generate process on [0, T]
            params
            :n (int) number of discretization on [0, T]
            :N (int) number of paths
            :method (str) 
                euler: Euler Scheme
        """
        paths = []
        for _ in range(N):
            dWt = np.random.normal(0, np.sqrt(T/n), n)
            for j in range(n):
                if method == 'euler':
                    if j == 0:
                        path = [self.S0]
                    St = path[j]
                    St_plus = St + self.a*St*T/n + self.b*np.power(St, self.gamma)*dWt[j]
                    path.append(St_plus)
            paths.append(path)
        return paths

if __name__ == "__main__":
    T = 1
    r = 0.01
    a = 0.1
    S0 = 100
    gamma = 2 
    b = 0.2/ np.power(S0, gamma-1)
    N = T * 2**6
    M = 1_000_000
    cev = CEVProcess(S0, a, b, gamma)
    np.random.seed(seed=0)
    euler_paths = cev.generate(1, 252, 1, 'euler')
    print(euler_paths[0][:5])
    print(doss_paths[0][:5])
    print(milstein_paths[0][:5])
