import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from active_preference import ActivePreference, RBF

def sample1D(x):
    """
1D sample function
    """
    return 0.3*np.exp(-(x-0.3)**2/0.01) + np.exp(-(x-0.7)**2/0.01)

def sample2D(x):
    """
2D sample function
    """
    x = x.reshape(-1,2)

    x1 = x[:,0]
    x2 = x[:,1]
    y = np.sin(x1)+x1/3+np.sin(12*x1)+np.sin(x2)+x2/3+np.sin(12*x2)-1
    return np.max([np.zeros(y.shape),y],axis=0)

def test1D():
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

    #f = np.zeros((5,5))
    #f[:,0] = np.arange(-2,3)
    #dpost = np.array([model._d_log_posterior(f[i])[0] for i in range(5)])
    #g = np.zeros((4,5))
    #g[:,0] = np.arange(-1.5,2)
    #ddpost = [model._dd_log_posterior(g[i])[0,0] for i in range(4)]
    #plt.plot(g[:,0], dpost[1:]-dpost[:-1])
    #plt.plot(g[:,0], ddpost, "x")
    ##plt.show()
    #plt.clf()
    #x = x.flatten()
    #plt.plot(x,fopt)
    #plt.plot(x,v(x.reshape(-1,1)))
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

    v = sample1D
    for i in range(10):
        x0,x1 = model.query()
        if v(x0) > v(x1): model.prefer(0)
        else: model.prefer(1)
        x = np.linspace(0,1,50).reshape(-1,1)
        mean = model._mean(x)
        sd = model._sd(x)
        plt.clf()
        plt.plot(x,mean)
        plt.plot(x,mean+sd)
        plt.plot(x,mean-sd)
        plt.plot(model._x,model._fMAP,"ro")
        plt.plot(x, model._expected_improvement(x))
        plt.show()

def test2D():
    model = ActivePreference([[0,1],[0,1]])

    v = sample2D
    x,y = np.meshgrid(np.linspace(0,1,50),np.linspace(0,1,50))
    w = np.array([x.flatten(),y.flatten()]).T
    z = v(w).reshape(50,50)
    #plt.clf()
    #plt.contour(x,y,z)
    #plt.show()

    for i in range(20):
        x0, x1 = model.query()
        if v(x0) > v(x1): model.prefer(0)
        else: model.prefer(1)
        plt.clf()
        plt.contour(x,y,model._expected_improvement(w).reshape(x.shape))
        plt.show()

    #mean = model._mean(w).reshape(x.shape)
    #plt.clf()
    #plt.contour(x,y,mean)
    #plt.show()

    #print(np.argmax(z), np.argmax(mean))

if __name__ == "__main__":
    test2D()
