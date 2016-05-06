# -*- coding: utf-8 -*-

from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)


class ACNavierStokesIntInters(BaseAdvectionDiffusionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, rhs, elemap, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver, c=self._tpl_c)

        self._be.pointwise.register('pyfr.solvers.acnavstokes.kernels.intconu')
        self._be.pointwise.register('pyfr.solvers.acnavstokes.kernels.intcflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'intconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal0_lhs, urin=self._scal0_rhs,
            ulout=self._vect0_lhs, urout=self._vect0_rhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal0_lhs, ur=self._scal0_rhs,
            gradul=self._vect0_lhs, gradur=self._vect0_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class ACNavierStokesMPIInters(BaseAdvectionDiffusionMPIInters):
    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        super().__init__(be, lhs, rhsrank, rallocs, elemap, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver, c=self._tpl_c)

        self._be.pointwise.register('pyfr.solvers.acnavstokes.kernels.mpiconu')
        self._be.pointwise.register('pyfr.solvers.acnavstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'mpiconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal0_lhs, urin=self._scal0_rhs, ulout=self._vect0_lhs
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal0_lhs, ur=self._scal0_rhs,
            gradul=self._vect0_lhs, gradur=self._vect0_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class ACNavierStokesBaseBCInters(BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       c=self._tpl_c, bctype=self.type,
                       bccfluxstate=self.cflux_state)

        self._be.pointwise.register('pyfr.solvers.acnavstokes.kernels.bcconu')
        self._be.pointwise.register('pyfr.solvers.acnavstokes.kernels.bccflux')

        self.kernels['con_u'] = lambda: self._be.kernel(
            'bcconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal0_lhs, ulout=self._vect0_lhs,
            nlin=self._norm_pnorm_lhs, ploc=self._ploc
        )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal0_lhs, gradul=self._vect0_lhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs,
            ploc=self._ploc
        )


class ACNavierStokesNoSlptWallBCInters(ACNavierStokesBaseBCInters):
    type = 'no-slp-wall'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tpl_c['v'] = self._eval_opts('uvw'[:self.ndims], default='0')


class ACNavierStokesSlpWallBCInters(ACNavierStokesBaseBCInters):
    type = 'slp-wall'
    cflux_state = None


class ACNavierStokesInflowBCInters(ACNavierStokesBaseBCInters):
    type = 'ac-in-fv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc, self._ploc = self._exp_opts(
            ['u', 'v', 'w'][:self.ndims], lhs
        )

        self._tpl_c.update(tplc)


class ACNavierStokesOutflowBCInters(ACNavierStokesBaseBCInters):
    type = 'ac-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc, self._ploc = self._exp_opts(
            ['p'], lhs
        )

        self._tpl_c.update(tplc)
