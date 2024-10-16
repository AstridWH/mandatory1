import numpy as np
import sympy as sp
import scipy.sparse as sparse
from scipy.interpolate import interpn

x, y = sp.symbols('x,y')

class Poisson2D:
    """Solve Poisson's equation in 2D::

        \nabla^2 u(x, y) = f(x, y), in [0, L]^2

    where L is the length of the domain in both x and y directions.
    Dirichlet boundary conditions are used for the entire boundary.
    The Dirichlet values depend on the chosen manufactured solution.
    """

    def __init__(self, L, ue):
        """Initialize Poisson solver for the method of manufactured solutions

        Parameters
        ----------
        L : number
            The length of the domain in both x and y directions
        ue : Sympy function
            The analytical solution used with the method of manufactured solutions.
            ue is used to compute the right hand side function f.
        """
        self.L = L
        self.ue = ue
        self.f = sp.diff(self.ue, x, 2)+sp.diff(self.ue, y, 2)

    def create_mesh(self, N):
        """Create 2D mesh and store in self.xij and self.yij

        Parameters
        ----------
        N : int
            number of steps in the domain
        """
        self.h = self.L/N
        xij = np.linspace(0, self.L, N+1)
        yij = np.linspace(0, self.L, N+1)
        mesh = np.meshgrid(xij, yij, indexing='ij')
        self.xij, self.yij = mesh
        self.N = N

    def D2(self):
        """Return second order differentiation matrix

        Returns
        -------
        D: array
            Second order differentiation matrix
        """
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    def laplace(self):
        """Return vectorized Laplace operator

        Return
        A: array
            the left hand side in Ax = b
        """
        D2x = (1. / self.h ** 2) * self.D2()
        D2y = (1. / self.h**2) * self.D2()
        A = (sparse.kron(D2x, sparse.eye(self.N + 1)) +
                sparse.kron(sparse.eye(self.N + 1), D2y))
        return A

    def get_boundary_indices(self):
        """Return indices of vectorized matrix that belongs to the boundary

        Retrun
        ------
        bnds: array
            contains indexces of the boundary
        """
        B = np.ones((self.N + 1, self.N + 1), dtype=bool)
        B[1:-1, 1:-1] = 0
        bnds = np.where(B.ravel() == 1)[0]
        return bnds

    def assemble(self):
        """Return assembled matrix A and right hand side vector b

        Return
        ------
        A: Array
            the left side in Ax = b
        b: Array
            the right side in Ax = b
        """
        A = self.laplace()
        bnds = self.get_boundary_indices()
        A = A.tolil()
        for i in bnds:
            A[i] = 0
            A[i, i] = 1
        A = A.tocsr()

        F = sp.lambdify((x, y), self.f)(self.xij, self.yij).ravel()
        b = F.ravel()
        fij = sp.lambdify((x, y), self.ue)(self.xij, self.yij).ravel()
        b[bnds] = fij[bnds]

        return A, b

    def l2_error(self, u):
        """Return l2-error norm

        Retrun
        ------
        L2: float
            the l2-error norm
        """
        ue = sp.lambdify((x, y), self.ue)(self.xij, self.yij)
        um = u - ue
        l2 = np.sqrt(np.sum((um) ** 2) * self.h **2)
        return l2

    def __call__(self, N):
        """Solve Poisson's equation.

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction

        Returns
        -------
        U: array
            The solution as a Numpy array
        """
        self.create_mesh(N)
        A, b = self.assemble()
        self.U = sparse.linalg.spsolve(A, b.flatten()).reshape((N+1, N+1))
        return self.U

    def convergence_rates(self, m=6):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretization levels to use

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            u = self(N0)
            E.append(self.l2_error(u))
            h.append(self.h)
            N0 *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

    def eval(self, x, y):
        """Return u(x, y)

        Parameters
        ----------
        x, y : numbers
            The coordinates for evaluation

        Returns
        -------
        The value of u(x, y)

        """
        coord = np.linspace(0, self.L, self.N+1) #this is just 1d x and y
        val = interpn((coord, coord), self.U, np.array([x, y]))[0]
        return val

def test_convergence_poisson2d():
    # This exact solution is NOT zero on the entire boundary
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    r, E, h = sol.convergence_rates()
    assert abs(r[-1]-2) < 1e-2

def test_interpolation():
    ue = sp.exp(sp.cos(4*sp.pi*x)*sp.sin(2*sp.pi*y))
    sol = Poisson2D(1, ue)
    U = sol(100)
    assert abs(sol.eval(0.52, 0.63) - ue.subs({x: 0.52, y: 0.63}).n()) < 1e-3
    assert abs(sol.eval(sol.h/2, 1-sol.h/2) - ue.subs({x: sol.h/2, y: 1-sol.h/2}).n()) < 1e-3

if __name__ == "__main__":
    test_convergence_poisson2d()
    test_interpolation()