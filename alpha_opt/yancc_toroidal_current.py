import yancc
from yancc.field import Field
from yancc.solve import solve_dke
from yancc.species import Hydrogen, Electron, GlobalMaxwellian
from yancc.velocity_grids import MaxwellSpeedGrid, UniformPitchAngleGrid

import jax, jax.numpy as jnp
import numpy as np

from interpax import CubicSpline

class ToroidalCurrentYANCC:
    def __init__(self, ne, Te, ni, Ti, ns=8, nt=51, nz=49, na=55, nx=6, rtol=1e-5, warm_restarts = False):
        self.ne = ne
        self.Te = Te
        self.ni = ni        
        self.Ti = Ti
        self.nt = nt
        self.nz = nz
        self.na = na
        self.nx = nx
        self.ns = ns
        self.rtol = rtol
        self.warm_restarts = warm_restarts

        # translate profiles to functions of rho
        s = jnp.linspace(0,1,50)
        rho = jnp.sqrt(s)
        self.Ti_rho = CubicSpline(rho, self.Ti(s))
        self.Te_rho = CubicSpline(rho, self.Te(s))
        self.ni_rho = CubicSpline(rho, self.ni(s))
        self.ne_rho = CubicSpline(rho, self.ne(s))
        
        
    def toroidal_current(self, vmec):
        # Check out https://terpconnect.umd.edu/~mattland/assets/notes/computing_vmec_AC_profile_from_a_bootstrap_current_code.pdf
        vmec.run()
        
        s=jnp.linspace(0.05, 0.95, self.ns)
        rho_=jnp.sqrt(s)

        globals_ = [
            GlobalMaxwellian(Hydrogen, self.Ti_rho, self.ni_rho),
            GlobalMaxwellian(Electron, self.Te_rho, self.ne_rho),
        ]
        
        pitchgrid = UniformPitchAngleGrid(self.na)
        speedgrid = MaxwellSpeedGrid(self.nx)

        fsa_j_dot_B = jnp.zeros(self.ns)
        fsa_B2 = jnp.zeros(self.ns)
        I = jnp.zeros(self.ns)
        # needs a filename to feed into yancc
        wout_filename = 'wout_nc_tmp.nc'
        vmec.wout.save(wout_filename)
        for i,rho in enumerate(rho_):
            field = Field.from_vmec(wout_filename, rho, self.nt, self.nz)

            fsa_B2 = fsa_B2.at[i].set(field.B2mag_fsa)
            I = I.at[i].set(field.I)
            psi_edge = field.Psi
            species = [g.localize(rho) for g in globals_]

            # Radial electric field, in Volts. Erho = -dPhi/drho.
            #Er_kV_per_m = 0.01
            Erho = 0.01 #Er_kV_per_m * field.a_minor * 1000

            if i>0 and self.warm_restarts:
                # using solution at previous radius as warm start
                # YOU PROBABLY DO NOT WANT TO USE THIS
                # TESTING HAS FOUND IT IS USUALLY SLOW THINGS DOWN AND/OR LEADS TO LARGER RESIDUALS
                # This was for ns=8. It probably works better for finer grids
                # but we do not want run on too fine a grid in optimization
                # because we can just interpolate
                sol, info = solve_dke(
                    field,
                    pitchgrid,
                    speedgrid,
                    species,
                    Erho=Erho,
                    verbose=2,
                    rtol=self.rtol,
                    f1=sol.f1
                )
            else:
                # cold start
                sol, info = solve_dke(
                    field,
                    pitchgrid,
                    speedgrid,
                    species,
                    Erho=Erho,
                    verbose=2,
                    rtol=self.rtol,
                )                
                
            x = sol.get("<J||B>")
            jax.debug.print("<J||B> = {x}", x=x)
            fsa_j_dot_B = fsa_j_dot_B.at[i].set(x)

        PI = 3.141592653589793
        mu0 = 4 * PI * 1.0e-7
        e = 1.602176565e-19
        dPds = self.ne.dfds(s) * self.Te.f(s) + self.ne.f(s) * self.Te.dfds(s) + self.ni.dfds(s) * self.Ti.f(s) + self.ni.f(s) * self.Ti.dfds(s)
        dPds = dPds * e # convert temp from eV to J so P is in Pascal
        jax.debug.print("I ={y}", y=I)
        jax.debug.print("dPds ={y}", y=dPds)
        jax.debug.print("fsa_B2 ={y}", y=fsa_B2)
        jax.debug.print("Psi ={y}", y=psi_edge)
        jax.debug.print("fsa_j_dot_B ={y}", y=fsa_j_dot_B)
       
        toroidal_current =  psi_edge * fsa_j_dot_B * 2*PI/fsa_B2   - mu0 * I * dPds/fsa_B2
        jax.debug.print("toroidal_current={y}", y=toroidal_current)
        return np.array(toroidal_current), np.array(s)

if __name__ == "__main__":
    # This part of the script is mainly intended for testing purposes
    # it takes about 30 minutes to run on 1 GPU on NERSC perlmutter but this can be reduced by lowering 'ns' (number of radii).
    
    class SimsoptLikeProfile:
        """This class merely exists to mimic needed features of simsopt's profile polynomials without needing a simsopt import.
        It wraps a numpy polynomial to provide the f(s) and dfds(s) methods of the simsopt profiles, where 's' is the normalized toroidal flux."""
        def __init__(self,  numpy_polynomial):
            self.p = numpy_polynomial

        def f(self, s):
            return self.p(s)

        def dfds(self, s):
            return self.p.deriv(1)(s)

        def __call__(self,s):
            return self.f(s)
            
    from vmecpp.simsopt_compat import Vmec
    import sys
    from numpy.polynomial import Polynomial
    
    ne0 = 3e20 # m^{-3}
    Te0 = 15e3 # eV
    ne_poly = Polynomial(ne0 * np.array([1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 1.2]))
    Te_poly = Polynomial(Te0 * np.array([1.0, -2.0, 2.0, -1.0]))
    ne = SimsoptLikeProfile(ne_poly)
    Te = SimsoptLikeProfile(Te_poly)
    ni = ne
    Ti = Te
    tc = ToroidalCurrentYANCC(ne, Te, ni, Ti, ns=8)

    # should be VMEC input file, like
    # input.vmec
    vmec_input_file = sys.argv[1]

    print(vmec_input_file, flush=True)
    vmec = Vmec(vmec_input_file)
    
    torcur, s = tc.toroidal_current(vmec)
    print(s)
    print(torcur)
