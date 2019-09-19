import numpy as np

class Bayesian_regression():
    def __init__(self, w0, L0, sigma_e, latent_dim):
        self.latent_dim = latent_dim

        # prior
        self.w0_bar = w0
        self.L0 = L0
        self.Sigma_e = sigma_e

        # posterior (parallelized)
        self.wt_unnorm = np.matmul(self.L0, self.w0_bar)
        self.wt_bar = self.w0_bar.copy()
        self.Lt_inv = np.linalg.inv(self.L0)

        #
        self.sample_mvn()

    def update_posterior(self, phit, rt):
        # inverse precision
        denum = self.Sigma_e + np.einsum('bi,ij,bj->b', phit, self.Lt_inv, phit)
        denum = np.reciprocal(denum)

        num = np.einsum('ij,bj->bi', self.Lt_inv, phit)
        num = np.einsum('bi,bj->bij', num, num)

        self.Lt_inv = self.Lt_inv - np.einsum('b,bij->ij', denum, num)

        # mean
        self.wt_unnorm = self.wt_unnorm + 1./self.Sigma_e * np.einsum('bi,b->i', phit, rt).reshape(-1,1)

        self.wt_bar = np.matmul(self.Lt_inv, self.wt_unnorm)

    def predict(self, phit):
        # predictive
        yt = np.einsum('ik,bia->ba', self.wt, phit)
        #Sigma_t = self.Sigma_e* (1.+ np.einsum('bi,ij,bj->b', phit, self.Lt_inv, phit))

        return yt#, Sigma_t

    def sample_mvn(self):
        z = np.random.normal(size=[self.latent_dim, 1])
        V, U = np.linalg.eigh(self.Lt_inv)
        self.wt = self.wt_bar + np.matmul(np.matmul(U, np.sqrt(np.diag(V))), z)