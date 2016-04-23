# -*- coding: utf-8 -*-

from pyfr.backends.base.generator import BaseKernelGenerator


class CUDAKernelGenerator(BaseKernelGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Specialise
        if self.ndim == 1:
            self._limits = 'if (_x < _nx)'
        else:
            self._limits = 'for (int _y = 0; _y < _ny && _x < _nx; ++_y)'

    def render(self):
        # Kernel spec
        spec = self._render_spec()

        # Iteration limits (if statement/for loop)
        limits = self._limits

        # Combine
        return '''{spec}
               {{
                   int _x = blockIdx.x*blockDim.x + threadIdx.x;
                   {limits}
                   {{
                       {body}
                   }}
               }}'''.format(spec=spec, limits=limits, body=self.body)

    def _render_spec(self):
        # We first need the argument list; starting with the dimensions
        kargs = ['int ' + d for d in self._dims]

        # Now add any scalar arguments
        kargs.extend('{0.dtype} {0.name}'.format(sa) for sa in self.scalargs)

        # Finally, add the vector arguments
        for va in self.vectargs:
            # Views
            if va.isview:
                kargs.append('{0.dtype}* __restrict__ {0.name}_v'.format(va))
                kargs.append('const int* __restrict__ {0.name}_vix'
                             .format(va))

                if va.ncdim >= 1:
                    kargs.append('const int* __restrict__ {0.name}_vcstri'
                                 .format(va))
                if va.ncdim == 2:
                    kargs.append('const int* __restrict__ {0.name}_vrstri'
                                 .format(va))
            # Arrays
            else:
                # Intent in arguments should be marked constant
                const = 'const' if va.intent == 'in' else ''

                kargs.append('{0} {1.dtype}* __restrict__ {1.name}_v'
                             .format(const, va).strip())

                if self.needs_lsdim(va):
                    kargs.append('int lsd{0.name}'.format(va))

        return '__global__ void {0}({1})'.format(self.name, ', '.join(kargs))
