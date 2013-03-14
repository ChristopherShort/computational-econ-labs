from scipy import interp

class LinInterp:
    """Provides linear interpolation in one dimension."""

    def __init__(self, grid, vals):
        """Attributes: 

            grid: array-like containing the grid points
            vals: array-like containing the values of a 
                  function at the grid points.
        
        """
        self.grid, self.vals = grid, vals

    def set_vals(self, new_vals):
        """Updates the function values."""
        self.vals = new_vals

    def __call__(self, z):
        """Basic linear interpolation.  If called on 
        a value z outside the original interpolation grid
        interp returns:

            min(val) if z < min(grid)
            max(val) if z > max(grid)

        """
        return interp(z, self.grid, self.vals)


