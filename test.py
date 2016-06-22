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

    x = np.linspace(0,1,5).reshape(-1,1)
    i = []
    j = []
    def v(x):
        if x.ndim==2:
            return x[:,0]
        else:
            return x[0]

    v = gaus_mix

    for a,b,k in zip(x[:-1],x[1:],range(len(x))):
        if v(a) > v(b):
            i.append(k)
            j.append(k+1)
        else:
            i.append(k+1)
            j.append(k)

    model._i = i
    model._j = j
    model.x = x

#    f = np.zeros((5,5))
#    f[:,0] = np.arange(-2,3)
#
#    dpost = np.array([model._d_log_posterior(f[i])[0] for i in range(5)])
#
#    g = np.zeros((4,5))
#    g[:,0] = np.arange(-1.5,2)
#
#    ddpost = [model._dd_log_posterior(g[i])[0,0] for i in range(4)]
#
#    plt.plot(g[:,0], dpost[1:]-dpost[:-1])
#    plt.plot(g[:,0], ddpost, "x")
#    #plt.show()
#    plt.clf()
#
#    fopt = model._argmax_posterior()
#    x = x.flatten()
#    plt.plot(x,fopt)
#    plt.plot(x,v(x.reshape(-1,1)))
#    plt.show()

    lng0 = np.linspace(0,5,10)
    lng1 = (lng0[1:]+lng0[:-1])/2
    ml = np.array([model._log_marginal_likelihood(1,y) for y in lng0])
    dml = np.array([model._d_log_marginal_likelihood_lng(1,y) for y in lng1])

    plt.clf()
    plt.plot(lng0,ml)
    plt.show()

    print(ml)
    print(dml)

    plt.clf()
    plt.plot(lng1, ml[1:]-ml[:-1])
    plt.plot(lng1, dml)
    plt.show()


if __name__ == "__main__":
    test()
