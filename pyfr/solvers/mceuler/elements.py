import numpy as np

from pyfr.solvers.baseadvec import BaseAdvectionElements
from pyfr.multicomp.mcfluid import MCFluid


class BaseMCFluidElements:
    @staticmethod
    def privars(ndims, cfg):
        species_names = MCFluid.get_species_names(cfg)[0:-1]

        if ndims == 2:
            return ['p', 'u', 'v', 'T'] + species_names
        elif ndims == 3:
            return ['p', 'u', 'v', 'w', 'T'] + species_names

    @staticmethod
    def convars(ndims, cfg):
        species_names = MCFluid.get_species_names(cfg)[0:-1]
        if ndims == 2:
            return ['rho', 'rhou', 'rhov', 'E'] + species_names
        elif ndims == 3:
            return ['rho', 'rhou', 'rhov', 'rhow', 'E'] + species_names

    dualcoeffs = convars

    @staticmethod
    def visvars(ndims, cfg):
        species_names = MCFluid.get_species_names(cfg)[0:-1]
        if ndims == 2:
            varmap = {
                'pressure': ['p'],
                'velocity': ['u', 'v'],
                'temperature': ['T']
            }
        elif ndims == 3:
            varmap = {
                'pressure': ['p'],
                'velocity': ['u', 'v', 'w'],
                'temperature': ['T']
            }
        for sn in species_names:
            varmap[sn] = [sn]

        return varmap

    @staticmethod
    def pri_to_con(pris, cfg):

        fluid = MCFluid(cfg)
        ns = fluid.consts['ns']
        ndims = len(pris) - (ns - 1) - 2
        consts = fluid.consts

        # Compute ns species
        Yns = 1.0 - sum(pris[ndims+2::])
        # Compute mixture properties
        Rmix = 0.0
        cp = 0.0
        for k,Y in enumerate(pris[ndims+2::]+[Yns]):
            Rmix += Y/consts['MW'][k]
            cp += Y*consts['cp0'][k]
        Rmix *= consts['Ru']

        # Compute density
        p, T = pris[0], pris[ndims + 1]
        rho = p/(Rmix*T)

        # Multiply velocity components by rho
        rhovs = [rho * c for c in pris[1 : ndims + 1]]

        # Compute the total energy
        rhok = 0.5 * rho * sum(c * c for c in pris[1 : ndims + 1])
        rhoe = rho*T*(cp-Rmix)
        rhoE = rhoe + rhok

        # Species mass
        rhoYk = [rho * c for c in pris[ndims + 2::]]

        return [rho, *rhovs, rhoE, *rhoYk]

    @staticmethod
    def con_to_pri(cons, cfg):
        fluid = MCFluid(cfg)
        ns = fluid.consts['ns']
        ndims = len(cons)-(ns-1)-2
        consts = fluid.consts

        rho, rhoE = cons[0], cons[ndims + 1]

        # Divide momentum components by rho
        vs = [rhov / rho for rhov in cons[1 : ndims + 1]]

        # Species Mass Fraction
        Yk = [rhoY / rho for rhoY in cons[ndims + 2 ::]]

        # Compute ns species
        Yns = 1.0 - sum(Yk)
        # Compute mixture properties
        Rmix = 0.0
        cp = 0.0
        for k,Y in enumerate(Yk+[Yns]):
            Rmix += Y/consts['MW'][k]
            cp += Y*consts['cp0'][k]
        Rmix *= consts['Ru']

        # Compute the temperature, pressure
        e = rhoE/rho - 0.5 *sum(v * v for v in vs)
        T = e/(cp-Rmix)
        p = rho*Rmix*T

        return [p, *vs, T, *Yk]

    @staticmethod
    def diff_con_to_pri(cons, diff_cons, cfg):
        rho, *rhouvw = cons[:-1]
        diff_rho, *diff_rhouvw, diff_E = diff_cons

        # Divide momentum components by ρ
        uvw = [rhov / rho for rhov in rhouvw]

        # Velocity gradients: ∂u⃗ = 1/ρ·[∂(ρu⃗) - u⃗·∂ρ]
        diff_uvw = [(diff_rhov - v*diff_rho) / rho
                    for diff_rhov, v in zip(diff_rhouvw, uvw)]

        # Pressure gradient: ∂p = (γ - 1)·[∂E - 1/2*(u⃗·∂(ρu⃗) + ρu⃗·∂u⃗)]
        gamma = cfg.getfloat('constants', 'gamma')
        diff_p = diff_E - 0.5*(sum(u*dru for u, dru in zip(uvw, diff_rhouvw)) +
                               sum(ru*du for ru, du in zip(rhouvw, diff_uvw)))
        diff_p *= gamma - 1

        return [diff_rho, *diff_uvw, diff_p]

    @staticmethod
    def validate_formulation(ctrl):
        shock_capturing = ctrl.cfg.get('solver', 'shock-capturing', 'none')
        if shock_capturing == 'entropy-filter':
            if ctrl.formulation == 'dual':
                raise ValueError('Entropy filtering not compatible with '
                                 'dual time stepping.')
            elif ctrl.controller_has_variable_dt:
                raise ValueError('Entropy filtering not compatible with '
                                 'adaptive time stepping.')

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Can elide shock-capturing at p = 0
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')

        # Modified entropy filtering method using specific physical
        # entropy (without operator splitting for Navier-Stokes)
        # doi:10.1016/j.jcp.2022.111501
        if shock_capturing == 'entropy-filter' and self.basis.order != 0:
            self._be.pointwise.register(
                'pyfr.solvers.mceuler.kernels.entropylocal'
            )
            self._be.pointwise.register(
                'pyfr.solvers.mceuler.kernels.entropyfilter'
            )

            # Template arguments
            eftplargs = {
                'ndims': self.ndims,
                'nupts': self.nupts,
                'nfpts': self.nfpts,
                'nvars': self.nvars,
                'nfaces': self.nfaces,
                'c': self.cfg.items_as('constants', float),
                'order': self.basis.order
            }

            # Check to see if running anti-aliasing
            if self.antialias:
                raise ValueError('Entropy filter not compatible with '
                                 'anti-aliasing.')

            # Check to see if running collocated solution/flux points
            m0 = self.basis.m0
            mrowsum = np.max(np.abs(np.sum(m0, axis=1) - 1.0))
            if np.min(m0) < -1e-8 or mrowsum > 1e-8:
                raise ValueError('Entropy filter requires flux points to be a '
                                 'subset of solution points or a convex '
                                 'combination thereof.')

            # Minimum density/pressure constraints
            eftplargs['d_min'] = self.cfg.getfloat('solver-entropy-filter',
                                                   'd-min', 1e-6)
            eftplargs['p_min'] = self.cfg.getfloat('solver-entropy-filter',
                                                   'p-min', 1e-6)

            # Entropy tolerance
            eftplargs['e_tol'] = self.cfg.getfloat('solver-entropy-filter',
                                                   'e-tol', 1e-6)

            # Hidden kernel parameters
            eftplargs['f_tol'] = self.cfg.getfloat('solver-entropy-filter',
                                                   'f-tol', 1e-4)
            eftplargs['niters'] = self.cfg.getfloat('solver-entropy-filter',
                                                    'niters', 20)
            efunc = self.cfg.get('solver-entropy-filter', 'e-func',
                                 'numerical')
            eftplargs['e_func'] = efunc
            if efunc not in {'numerical', 'physical'}:
                raise ValueError(f'Unknown entropy functional: {efunc}')

            # Precompute basis orders for filter
            ubdegs = self.basis.ubasis.degrees
            eftplargs['ubdegs'] = [int(max(dd)) for dd in ubdegs]
            eftplargs['order'] = self.basis.order

            # Compute local entropy bounds
            self.kernels['local_entropy'] = lambda uin: self._be.kernel(
                'entropylocal', tplargs=eftplargs, dims=[self.neles],
                u=self.scal_upts[uin], entmin_int=self.entmin_int
            )

            # Apply entropy filter
            self.kernels['entropy_filter'] = lambda uin: self._be.kernel(
                'entropyfilter', tplargs=eftplargs, dims=[self.neles],
                u=self.scal_upts[uin], entmin_int=self.entmin_int,
                vdm=self.vdm, invvdm=self.invvdm
            )


class MCEulerElements(BaseMCFluidElements, BaseAdvectionElements):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mcfluid = MCFluid(self.cfg)

    def set_backend(self, *args, **kwargs):
        super().set_backend(*args, **kwargs)

        # Can elide interior flux calculations at p = 0
        if self.basis.order == 0:
            return

        # Register our flux kernels
        self._be.pointwise.register('pyfr.solvers.mceuler.kernels.tflux')

        c = self.cfg.items_as('constants', float)
        c |= self.mcfluid.consts
        # Template parameters for the flux kernels
        tplargs = {
            'ndims': self.ndims,
            'nvars': self.nvars,
            'nverts': len(self.basis.linspts),
            'c': c,
            'eos': self.mcfluid.eos,
            'jac_exprs': self.basis.jac_exprs
        }

        # Helpers
        tdisf = []
        c, l = 'curved', 'linear'
        r, s = self._mesh_regions, self._slice_mat
        slicedk = self._make_sliced_kernel

        if c in r and 'flux' not in self.antialias:
            tdisf.append(lambda uin: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nupts, r[c]], u=s(self.scal_upts[uin], c),
                f=s(self._vect_upts, c), smats=self.curved_smat_at('upts')
            ))
        elif c in r:
            tdisf.append(lambda: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'curved'},
                dims=[self.nqpts, r[c]], u=s(self._scal_qpts, c),
                f=s(self._vect_qpts, c), smats=self.curved_smat_at('qpts')
            ))

        if l in r and 'flux' not in self.antialias:
            tdisf.append(lambda uin: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'linear'},
                dims=[self.nupts, r[l]], u=s(self.scal_upts[uin], l),
                f=s(self._vect_upts, l), verts=self.ploc_at('linspts', l),
                upts=self.upts
            ))
        elif l in r:
            tdisf.append(lambda: self._be.kernel(
                'tflux', tplargs=tplargs | {'ktype': 'linear'},
                dims=[self.nqpts, r[l]], u=s(self._scal_qpts, l),
                f=s(self._vect_qpts, l), verts=self.ploc_at('linspts', l),
                upts=self.qpts
            ))

        if 'flux' not in self.antialias:
            self.kernels['tdisf'] = lambda uin: slicedk(k(uin) for k in tdisf)
        else:
            self.kernels['tdisf'] = lambda: slicedk(k() for k in tdisf)
