import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from active_preference import ActivePreference

def gaus_mix(x):
    """
    sample function
    """
    return 0.3*np.exp(-(x-0.3)**2/0.01) + np.exp(-(x-0.7)**2/0.01)

def test():
    model = ActivePreference([[0,1]])

    x = list(np.linspace(0,1,5).reshape(-1,1))
    i = []
    j = []
    def v(x):
        if x.ndim==2:
            return -x[:,0]
        else:
            return -x[0]

    v = gaus_mix

    for a,b,k in zip(x[:-1],x[1:],range(len(x))):
        if v(a) > v(b):
            i.append(k)
            j.append(k+1)
        else:
            i.append(k+1)
            j.append(k)

    model._x = x
    model._i = i
    model._j = j

    model._kernel = model._kernel_fun(np.array(model._x).reshape(-1,1))

    f = np.zeros((5,5))
    f[:,0] = np.arange(-2,3)

    dpost = np.array([model._d_log_likelihood(f[i])[0] for i in range(5)])

    g = np.zeros((4,5))
    g[:,0] = np.arange(-1.5,2)

    ddpost = [model._dd_log_likelihood(g[i])[0,0] for i in range(4)]

    plt.plot(g[:,0], dpost[1:]-dpost[:-1])
    plt.plot(g[:,0], ddpost, "x")
    plt.show()
    plt.clf()

    fopt = model._maximize_posterior()
    x = np.array(x).flatten()
    plt.plot(x,fopt)
    plt.plot(x,v(x.reshape(-1,1)))
    plt.show()


if __name__ == "__main__":
    test()
