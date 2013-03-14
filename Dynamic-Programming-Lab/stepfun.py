import numpy as np

class StepFun:

    def __init__(self, X, Y):
        """Creates an instance of the StepFun class.

        Parameters: 

            X: an increasing array of length n
            Y: an array of length n

        Implements a step function s, where 
        
            s(x) = sum_{i=0}^{n-1} Y[i] 1{X[i] <= x < X[i+1]}

        where X[n] := infty

        """
        self.X, self.Y = X, Y

    def __call__(self, x):
        """Evaluate the step function at x."""
        if x < self.X[0]:
            return 0.0
        i = self.X.searchsorted(x, side='right') - 1
        return self.Y[i]

    def expectation(self, F):
        """Computes expectation of s(Y) when F is the cdf of Y."""
        probs = np.append(F(self.X), 1)
        return np.dot(self.Y, probs[1:] - probs[:-1]) 
