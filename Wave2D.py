import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        xij = np.linspace(0, 1, N + 1)
        yij = np.linspace(0, 1, N + 1)
        mesh = np.meshgrid(xij, yij, indexing='ij')
        self.xij, self.yij = mesh
        self.N = N

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), 'lil')
        #D[0, :4] = 2, -5, 4, -1
        #D[-1, -4:] = -1, 4, -5, 2
        D[0] = 0 #why are these 0? who knows. but I isolated error to here and was testing out.
        D[-1] = 0
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        self.kx = self.mx*np.pi
        self.ky = self.my*np.pi
        w = self.c * np.sqrt(self.kx**2 + self.ky**2)
        return w

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """

        D = self.D2(N) / self.h ** 2

        U0 = sp.lambdify((x, y, t), self.ue(mx, my)) (self.xij, self.yij, 0)
        U1 = U0[:] + 0.5*(self.c*self.dt)**2*(D @ U0 + U0 @ D.T)

        return U0, U1


    @property
    def dt(self):
        """Return the time step"""
        dt = self.cfl*self.h/self.c
        return dt

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0)
        um = u - ue
        l2 = np.sum((um) ** 2) * self.h ** 2
        #print(type(l2))
        l2 = np.sqrt(l2)
        return l2

    def apply_bcs(self):
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.cfl = cfl
        self.c = c
        self.h = 1/N # L=1
        self.mx = mx
        self.my = my

        self.create_mesh(N)
        Unp1, Un, Unm1 = np.zeros((3, N + 1, N + 1))
        U0, U1 = self.initialize(N, mx, my)
        Unm1[:] = U0
        Un[:] = U1
        D = self.D2(N) / self.h ** 2

        plotdata = {0: Unm1.copy()}
        if store_data == 1:
            plotdata[1] = Un.copy()
        for n in range(1, Nt):
            Unp1[:] = 2 * Un - Unm1 + (c * self.dt) ** 2 * (D @ Un + Un @ D.T)
            # Set boundary conditions
            self.Unp1 = Unp1
            self.apply_bcs()
            Unp1 = self.Unp1
            # Swap solutions
            Unm1[:] = Un
            Un[:] = Unp1
            if n % store_data == 0:
                plotdata[n] = Unm1.copy()  # Unm1 is now swapped to Un

        if store_data > 0:
            return plotdata
        elif store_data == -1:
            type(self.h)
            l2 = self.l2_error(Un, Nt * self.dt)
            type(l2)
            return self.h, l2

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

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
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err)
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)




class Wave2D_Neumann(Wave2D):
    def D2(self, N):
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), 'lil')
        D[0, :2] = -2, 2 #unsure on these, tried to math it but made little sense
        D[-1, -2:] = 2, -2
        return D

    def ue(self, mx, my):
        ue = sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)
        return ue

    def apply_bcs(self):
        #?
        pass



def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05


def test_exact_wave2d():
    sol = Wave2D()
    h, l2 = sol(10, 10, mx=3, my=3, cfl=1/np.sqrt(2), store_data=-1)
    assert l2 < 10**(-12)

    print("1/2 wave2d")
    sol = Wave2D_Neumann()
    h, l2 = sol(10, 10, mx=3, my=3, cfl=1 / np.sqrt(2), store_data=-1)
    assert l2 < 10 ** (-12)

if __name__ == "__main__":
    test_convergence_wave2d()
    print("convergence")
    test_convergence_wave2d_neumann()
    print("convergence neumann")
    test_exact_wave2d()
    print("exact_wave2d")
    wave = Wave2D()
    data = wave(10, 10, store_data=1)

    print("let's animate")

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for n, val in data.items():
        frame = ax.plot_wireframe(wave.xij, wave.yij, val, rstride=2, cstride=2);
        # frame = ax.plot_surface(xij, yij, val, vmin=-0.5*data[0].max(),
        #                        vmax=data[0].max(), cmap=cm.coolwarm,
        #                        linewidth=0, antialiased=False)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                    repeat_delay=1000)
    ani.save('neumannwave.gif', writer='pillow', fps=5)

    print("done")