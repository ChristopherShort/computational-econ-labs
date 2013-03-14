import numpy as np

class SRS:
    
    def __init__(self, seed=None, F=None, phi=None, X=None):
        """Represents the model X_t = F(X_{t-1}, W_t), where W ~ phi

        Parameters: 
            seed: Seed value for random number generator.
            F:    Equation of motion for the state variable, X.
            phi:  A function returning a draw from some underlying 
                  probability distribution 
            X:    A number representing the initial condition of the 
                  state variable.
                  
        """
        self.F, self.phi, self.X = F, phi, X

        # set the seed for RNG
        if seed != None:
            self.set_seed(seed)
        
    def set_seed(self, seed):
        """Sets the seed for the random number generator."""
        self.seed = seed
        np.random.seed(self.seed)
        
    def update(self):
        """Updated the state according to X = F(X, Z)."""
        self.Z = self.phi()
        self.X = self.F(self.X, self.Z)

    def get_sample_path(self, T):
        """Generate path of length n from current state."""
        # check that seed is set
        if self.seed == None:
            raise Exception, "You need to provide a seed for the RNG!"

        # storage container
        path = np.zeros((T, 2))
        
        for t in xrange(T):
            path[t,0] = self.X
            path[t,1] = self.Z
            self.update()
        return path

    def get_marginal(self, init=None, T=None, N=None):
        """Returns N draws of X_T, starting from the state X=init."""
        # check that seed is set
        if self.seed == None:
            raise Exception, "You need to provide a seed for the RNG!"

        # storage container
        samples = np.zeros((N, 2))
                
        for i in xrange(N):
            self.X = init
            for t in xrange(T):
                self.update()
            samples[i, 0] = self.X
            samples[i, 1] = self.Z
            
        return samples
