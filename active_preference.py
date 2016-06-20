import numpy as np
import scipy as sp
from scipy.stats import norm
from sklearn.metrics.pairwise import pairwise_kernels

class ActivePreference():
    def __init__(self, bound):
        self._bound = np.array(bound)
        assert len(self._bound.shape) == 2
        assert self._bound.shape[1] == 2
        self._x = [self._bound.mean(axis=1)]
        self.dim = self._bound.shape[0]
        self._i = []
        self._j = []
        self._last_query = None
        self._sgm = 1.0

        self._kernel_fun = lambda x,y=None: pairwise_kernels(x,y,metric='rbf',gamma=10)

    def query(self):
        pass

    def answer(self, choice, query=None):

        if not query:
            query = self._last_query
        assert type(query) == list and len(query) == 2

        assert choice in [0,1]
        if choice == 0:
            p = query[0]
            q = query[1]
        else:
            p = query[1]
            q = query[0]

        if p not in self.x:
            self._x.append(p)

        if q not in self.x:
            self._x.append(q)

        self._i.append(_x.index(p))
        self._j.append(_x.index(q))

        self._kernel = self._kernel_fun(self._x)


    def best(self):
        pass

    def _log_likelihood(self,f):
        zk = (f[np.array(self._i)]-f[np.array(self._j)])/(np.sqrt(2)*self._sgm)
        return np.sum(np.log(norm.cdf(zk)))

    def _log_prior(self,f):
        alpha = np.linalg.solve(np.linalg.cholesky(self._kernel),f)
        return -0.5*np.sum(alpha**2)

    def _log_posterior(self, f):
        return self._log_likelihood(f) + self._log_prior(f)

    def _d_log_likelihood(self, f):
        k = len(self._i)
        zk = (f[np.array(self._i)]-f[np.array(self._j)])/(np.sqrt(2)*self._sgm)
        sik = np.zeros((len(f),k))
        sik[self._i,np.arange(k)] = 1/np.sqrt(2)*self._sgm
        sik[self._j,np.arange(k)] = -1/np.sqrt(2)*self._sgm
        return np.matmul(sik, norm.pdf(zk)/norm.cdf(zk))

    def _d_log_prior(self, f):
        return -np.linalg.solve(self._kernel,f)

    def _d_log_posterior(self, f):
        return self._d_log_likelihood(f) + self._d_log_prior(f)

    def _dd_log_likelihood(self, f):
        k = len(self._i)
        zk = (f[np.array(self._i)]-f[np.array(self._j)])/(np.sqrt(2)*self._sgm)
        sik = np.zeros((len(f),k))
        sik[self._i,np.arange(k)] =  1/np.sqrt(2)*self._sgm
        sik[self._j,np.arange(k)] = -1/np.sqrt(2)*self._sgm
        sijk = np.tensordot(sik,sik,axes=0)[:,np.arange(k),:,np.arange(k)].swapaxes(0,2)
        pdf = norm.pdf(zk)
        cdf = norm.cdf(zk)
        return np.tensordot(sijk, -pdf**2/cdf**2-zk*pdf/cdf,axes=([2],[0]))

    def _dd_log_prior(self, f):
        return np.linalg.inv(self._kernel)

    def __dd_log_posterior(self, f):
        return self._dd_log_likelihood(f) + self._dd_log_prior(f)

    def _maximize_posterior(self):
        def obj(f):
            return f - np.matmul(self._kernel, self._d_log_likelihood(f))

        def dobj(f):
            return np.identity(len(f)) -\
                np.matmul(self._kernel, self._dd_log_likelihood(f))

        return sp.optimize.fsolve(obj, np.zeros(len(self._x)), fprime=dobj)
