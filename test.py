import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from active_preference import ActivePreference, RBF

def gaus_mix(x):
    """
    sample function
    """
    return 0.3*np.exp(-(x-0.3)**2/0.01) + np.exp(-(x-0.7)**2/0.01)

def test():
    model = ActivePreference([[0,1]])

    #x = np.linspace(0,1,5).reshape(-1,1)
    #i = []
    #j = []
    #def v(x):
    #    if x.ndim==2:
    #        return x[:,0]
    #    else:
    #        return x[0]
    #v = gaus_mix
    #for a,b,k in zip(x[:-1],x[1:],range(len(x))):
    #    if v(a) > v(b):
    #        i.append(k)
    #        j.append(k+1)
    #    else:
    #        i.append(k+1)
    #        j.append(k)
    #model._u = i
    #model._v = j
    #model.x = x

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
    #    x = x.flatten()
    #    plt.plot(x,fopt)
    #    plt.plot(x,v(x.reshape(-1,1)))
    #    plt.show()

    #lng0 = np.linspace(-5,5,100)
    #lng1 = (lng0[1:]+lng0[:-1])/2
    #ml = np.array([model._log_marginal_likelihood(1,y) for y in lng0])
    #dml = np.array([model._d_log_marginal_likelihood_lng(1,y) for y in lng1])

    #plt.clf()
    #plt.plot(lng1, (ml[1:]-ml[:-1])/(lng0[1]-lng0[0]))
    #plt.plot(lng1, dml)
    ##plt.show()

    #lns0 = np.linspace(-5,5,100)
    #lns1 = (lns0[1:]+lns0[:-1])/2
    #ml = np.array([model._log_marginal_likelihood(y,-1) for y in lns0])
    #dml = np.array([model._d_log_marginal_likelihood_lns(y,-1) for y in lns1])

    #plt.clf()
    #plt.plot(lns1, (ml[1:]-ml[:-1])/(lns0[1]-lns0[0]))
    #plt.plot(lns1, dml)
    #plt.show()


    #model._maximize_marginal_likelihood()
    #xs = np.linspace(0,1,50).reshape(-1,1)
    #ei = model._expected_improvement(xs)
    #mean = model._mean(xs)
    #sd = model._sd(xs)
    #x_eiopt = model._argmax_expected_improvement()
    #plt.clf()
    #plt.plot(xs,mean)
    #plt.plot(xs,mean+sd)
    #plt.plot(xs,mean-sd)
    #plt.show()
    #plt.clf()
    #plt.plot(xs,ei)
    #plt.plot([x_eiopt],
    #         model._expected_improvement([x_eiopt]),"ro")
    #plt.show()

    v = gaus_mix
    for i in range(10):
        x0,x1 = model.query()
        if v(x0) > v(x1): model.prefer(0)
        else: model.prefer(1)
        xs = np.linspace(0,1,50).reshape(-1,1)
        mean = model._mean(xs)
        sd = model._sd(xs)
        plt.clf()
        plt.plot(xs,mean)
        plt.plot(xs,mean+sd)
        plt.plot(xs,mean-sd)
        plt.plot(model._x,model._fMAP,"ro")
        plt.plot(xs, model._expected_improvement(xs))
        plt.show()


if __name__ == "__main__":
    test()
