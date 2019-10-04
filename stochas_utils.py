import types
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar

class PoissonProcess():
    
    def __init__(self):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError

class HomoPoissonProcess(PoissonProcess):
    """
        Homogeneous Poisson Process
    """
    def __init__(self, lam):
        if type(lam) != int and type(lam) != float:
            raise ValueError(f"Expected lambda type: int or float, got {type(lam)}")
        self.lam = lam
    
    def generate(self, T, N, *, size=1, method='uniform', random_seed=None):
        """
            generate poisson process on [0, T] of fixed size
            N: number of discretization on [0, T]
            method: methods to generate realization jumping time 
                'uniform': conditional on NT = n, the jumping times are uniformly distributed
                'exponential': time interval between two jump times are exponentially distributed
        """
        # set random state
        np.random.seed(seed=random_seed)
        if method == 'uniform':
            # number of jumps
            n = np.random.poisson(self.lam*T)
            # realization of stopping times
            self.tau = np.sort(np.random.uniform(0, T, size=n))
        elif method == 'exponential':
            beta = 1 / self.lam # scale param
            t = 0
            taus = []
            while True: 
                tau = np.random.exponential(beta)
                t += tau
                if t < T:
                    taus.append(t)
                else:
                    break
            self.tau = np.sort(np.array(taus))
        else:
            raise ValueError(f"Expected method: uniform or exponential, got {method}")

        delta = T / N
        self.time_axis = np.arange(0, T, delta)
        self.Nt = np.zeros(N)
        for t in self.tau:
            self.Nt[self.time_axis > t] += size
        return self.Nt, self.tau

    def plot(self, *args, **kwargs):
        return plt.plot(self.time_axis, self.Nt, *args, **kwargs)

class InhomoPoissonProcess(PoissonProcess):
    """
        Homogeneous Poisson Process
    """
    def __init__(self, lam):
        if type(lam) != types.FunctionType:
            raise ValueError(f"Expected {types.FunctionType}, got {type(lam)}")
        self.lam = lam
    
    def generate(self, T, N, *, size=1, method='inverse', random_seed=None):
        """
            generate poisson process on [0, T] of fixed size
            N: number of discretization on [0, T]
            method:
                'inverse': inverse cdf method
        """
        # set random state
        np.random.seed(seed=random_seed)
        if method == 'inverse':
            t = 0
            taus = []
            while True: 
                u = np.random.uniform(0, 1)
                def func(T):
                    integral, err = quad(self.lam ,t, T)
                    return u - (1 - np.exp(-integral))
                try:
                    tau = root_scalar(func, bracket=(t,T))
                except ValueError:
                    break
                else:
                    t = tau.root
                    taus.append(t)
            self.tau = np.sort(np.array(taus))
        else:
            raise ValueError(f"Expected method: uniform or exponential, got {method}")

        delta = T / N
        self.time_axis = np.arange(0, T, delta)
        self.Nt = np.zeros(N)
        for t in self.tau:
            self.Nt[self.time_axis > t] += size
        return self.Nt, self.tau



if __name__ == "__main__": 
    
    pois_proc = HomoPoissonProcess(10)
    _, tau = pois_proc.generate(10, 10000, method='exponential')
    tau

    lam = lambda t: 20*np.exp(-(5-t)**2)
    pois_proc = InhomoPoissonProcess(lam)
    _, tau = pois_proc.generate(10, 10000)
    tau