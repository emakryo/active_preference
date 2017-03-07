import numpy as np
from scipy.optimize import basinhopping, fsolve
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances

class ActivePreference():
    """
Model for active preference learning.
    """
    def __init__(self, bound, sgm=0.01, gam=10, use_direct=True):
        try:
            import DIRECT
        except(ImportError):
            self.use_direct = False
        else:
            self.use_direct = use_direct

        self._bound = np.array(bound)
        assert len(self._bound.shape) == 2
        assert self._bound.shape[1] == 2
        assert sgm > 0
        assert gam > 0
        self._ndim = self._bound.shape[0]
        self._x = []
        self._u = []
        self._v = []
        self._last_query = None
        self._sgm = sgm
        self._kernel = RBF([[0]], gamma=gam)


    def query(self):
        """
Return most informative two points to compare
        """
        if len(self._x) == 0:
            #self._last_query = tuple(np.random.rand(2,self._ndim)*
            #                         (self._bound[:,1]-self._bound[:,0])+
            #                         self._bound[:,0])
            self._last_query = self._bound[:,0], self._bound[:,1]
        else:
            x_ie = self._argmax_expected_improvement()

            if np.any(np.sum((x_ie-np.array(self._x))**2,axis=1) < 1e-5):
                raise Exception("Maximum expected improvement is duplicated")

            x_max = self._x[np.argmax(self._fMAP)]
            self._last_query = x_ie, x_max

        return self._last_query

    def prefer(self, choice, query=None):
        """
Answer to the query. choice is 0 (first item) or 1 (second item).
If query is None, two points of the last query are compared.
        """
        if query is None:
            query = self._last_query
        assert len(query) == 2

        assert choice in [0,1]
        if choice == 0:
            p = query[0]
            q = query[1]
        else:
            p = query[1]
            q = query[0]

        pbool = [np.all(x == p) for x in self._x]
        qbool = [np.all(x == q) for x in self._x]

        if np.any(pbool):
            self._u.append(pbool.index(True))
        else:
            self._u.append(len(self._x))
            self._x.append(p)

        if np.any(qbool):
            self._v.append(qbool.index(True))
        else:
            self._v.append(len(self._x))
            self._x.append(q)

        self._kernel.X = self._x
        self._maximize_posterior()
        #self._maximize_marginal_likelihood()

    def best(self):
        return self._x[np.argmax(self._fMAP)]

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = np.array(x)
        self._kernel.X = self._x
        self._maximize_posterior()

    @property
    def lng(self):
        return np.log(self._kernel.gamma)

    @lng.setter
    def lng(self, lng):
        self._kernel.gamma = np.exp(lng)

    @property
    def lns(self):
        return np.log(self._sgm)

    @lns.setter
    def lns(self, lns):
        self._sgm = np.exp(lns)

    def _log_likelihood(self, f=None):
        if f is None: f = self._fMAP
        zk = (f[np.array(self._u)]-f[np.array(self._v)])/(np.sqrt(2)*self._sgm)
        return np.sum(np.log(norm.cdf(zk)))

    def _log_prior(self, f=None):
        if f is None: f = self._fMAP
        try:
            alpha = np.linalg.solve(np.linalg.cholesky(self._kernel()),f)
        except Exception as e:
            print(self._kernel())
            print(self._kernel._gamma)
            raise e

        return -0.5*np.sum(alpha**2)

    def _unnorm_log_posterior(self, f=None):
        if f is None: f = self._fMAP
        return self._log_likelihood(f) + self._log_prior(f)

    def _d_log_likelihood(self, f=None):
        if f is None: f = self._fMAP
        k = len(self._u)
        zk = (f[np.array(self._u)]-f[np.array(self._v)])/(np.sqrt(2)*self._sgm)
        sik = np.zeros((len(f),k))
        sik[self._u,np.arange(k)] = 1
        sik[self._v,np.arange(k)] = -1
        return np.matmul(sik, norm.pdf(zk)/norm.cdf(zk))/(np.sqrt(2)*self._sgm)

    def _d_log_prior(self, f=None):
        if f is None: f = self._fMAP
        return -np.linalg.solve(self._kernel(),f)

    def _d_unnorm_log_posterior(self, f=None):
        if f is None: f = self._fMAP
        return self._d_log_likelihood(f) + self._d_log_prior(f)

    def _dd_log_likelihood(self, f=None):
        if f is None: f = self._fMAP
        k = len(self._u)
        zk = (f[np.array(self._u)]-f[np.array(self._v)])/(np.sqrt(2)*self._sgm)
        # sik[i,k] = 1 if x[i] == x[u[k]]
        #          = -1 if x[i] == x[v[k]]
        #          = 0                otherwise
        sik = np.zeros((len(f),k))
        sik[self._u,np.arange(k)] =  1
        sik[self._v,np.arange(k)] = -1
        sijk = sik.reshape(-1,1,k)*sik.reshape(1,-1,k)
        pdf = norm.pdf(zk)
        cdf = norm.cdf(zk)
        return np.tensordot(sijk, -pdf**2/cdf**2-zk*pdf/cdf,axes=([2],[0]))/(2*self._sgm**2)

    def _argmax_posterior(self):
        def obj(f):
            return f - np.matmul(self._kernel(), self._d_log_likelihood(f))

        def dobj(f):
            return np.identity(len(f)) -\
                np.matmul(self._kernel(), self._dd_log_likelihood(f))

        return fsolve(obj, np.zeros(len(self._x)), fprime=dobj)

    def _maximize_posterior(self):
        self._fMAP = self._argmax_posterior()

    def _log_marginal_likelihood(self, lns, lng):
        self.lng = lng
        self.lns = lns
        self._maximize_posterior()
        m = np.identity(len(self._x))+np.matmul(self._kernel(),-self._dd_log_likelihood())
        s = self._unnorm_log_posterior()
        t = -0.5*np.log(np.linalg.det(m))
        return s+t

    def _argmax_marginal_likelihood(self, x0=None, n_iter=10):
        def obj(theta):
            return -self._log_marginal_likelihood(*theta)

        if x0 is None:
            x0 = [self.lns, self.lns]

        res = [sp.optimize.minimize(obj, x0)]
        res += [sp.optimize.minimize(obj, np.random.randn(2)*2) for _ in range(n_iter)]
        res = [r for r in res if r.success]
        if len(res):
            return min(res, key=lambda r: r.fun).x
        else:
            raise Exception(res.message)

    def _maximize_marginal_likelihood(self):
        theta_opt = self._argmax_marginal_likelihood()
        self.lng = theta_opt[0]
        self._sgm = np.exp(theta_opt[1])
        self._maximize_posterior()

    def _mean(self, x):
        ks = self._kernel(self._x,x)
        return np.matmul(ks.T, np.linalg.solve(self._kernel(),self._fMAP))

    def _sd(self, x):
        kss = self._kernel(x,x)
        ks = self._kernel(self._x,x)
        L = -self._dd_log_likelihood()
        #beta=np.linalg.solve(np.identity(len(self._x))+
        #                     np.matmul(L, self._kernel()),
        #                     np.matmul(L,ks))
        beta = np.linalg.solve(self._kernel(), ks)
        sd = np.sqrt(np.diag(kss - np.matmul(ks.T, beta)))
        sd[np.isnan(sd)] = 0
        return sd

    def _expected_improvement(self,x):
        mean_max = np.max(self._fMAP)
        mean = self._mean(x)
        sd = self._sd(x)
        ind = sd > 0
        cdf = np.zeros(mean.shape)
        cdf[ind] = norm.cdf((mean_max-mean[ind])/sd[ind])
        x_ei = (mean-mean_max)*cdf+sd*cdf
        #x_ei = (mean_max-mean)*cdf+sd*cdf
        return x_ei

    def _argmax_expected_improvement(self):
        def obj(x):
            return -self._expected_improvement([x])

        if self.use_direct:
        # use DIRECT algorithm
            x, _, _ = DIRECT.solve(obj, self._bound[:,0], self._bound[:,1],
                                     maxf=1000, algmethod=1)

        else:
            # use basinhopping
            res = basinhopping(obj, (self._bound[:,0]+self._bound[:,1]),
                    minimizer_kwargs={'bounds':self._bound})
            x = res.x

        return x


class RBF():
    def __init__(self, X, Y=None, gamma=None):
        self._X = np.array(X)

        if Y is None: self._Y = None
        else: self._Y = np.array(Y)

        if gamma is None: self._gamma = 1.0/self._X.shape[1]
        else: self._gamma = gamma

        self._K = self(self._X, self._Y, self._gamma)

    @property
    def X(self):
        return self._X
    @property
    def Y(self):
        return self._Y
    @property
    def gamma(self):
        return self._gamma

    @X.setter
    def X(self, X):
        self._X = X
        self._ssd = euclidean_distances(self._X, self._Y, squared=True)
        self._K = np.exp(-self._gamma*self._ssd)

    @Y.setter
    def Y(self, Y):
        self._Y = Y
        self._ssd = euclidean_distances(self._X, self._Y, squared=True)
        self._K = np.exp(-self._gamma*self._ssd)

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
        self._K = np.exp(-self._gamma*self._ssd)

    def __call__(self, X=None, Y=None, gamma=None):
        if X is None and Y is None and gamma is None: return self._K
        if X is None: X = self._X
        if Y is None: Y = self._Y
        if gamma is None: gamma = self._gamma
        assert gamma > 0
        ssd = euclidean_distances(X, Y, squared=True)
        return np.exp(-gamma*ssd)

    def d_gamma(self, gamma=None):
        if gamma is None: gamma = self._gamma
        return -self._ssd*np.exp(-gamma*self._ssd)
