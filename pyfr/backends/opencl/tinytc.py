from ctypes import (POINTER, Structure, addressof, byref, c_int, c_double,
                    c_float, c_size_t, c_uint, c_uint32, c_void_p)

import numpy as np

from pyfr.backends.opencl.provider import OpenCLKernel, OpenCLKernelProvider
from pyfr.ctypesutil import LibWrapper


# Possible TinyTC exception types
class TinyTCError(Exception): pass


class TinyTCMem(Structure):
    _fields_ = [
        ('value', c_void_p),
        ('type', c_int)
    ]


class TinyTCWrappers(LibWrapper):
    _libname = 'tinytc_cl'

    # Error codes
    _statuses = {
        '*': TinyTCError
    }

    # Constants
    MEM_TYPE_BUF = 0
    SCALAR_TYPE_F32 = 10
    SCALAR_TYPE_F64 = 11

    # Functions
    _functions = [
        (c_int, 'tinytc_cl_core_info_create', POINTER(c_void_p), c_void_p),
        (c_int, 'tinytc_cl_recipe_handler_submit', c_void_p, c_void_p,
         c_uint, POINTER(c_void_p), POINTER(c_void_p)),
        (c_int, 'tinytc_cl_recipe_handler_create', POINTER(c_void_p),
         c_void_p, c_void_p, c_void_p),
        (c_int, 'tinytc_recipe_tall_and_skinny_create', POINTER(c_void_p),
         c_void_p, c_int, c_uint32, c_uint32, c_uint32, c_void_p),
        (c_int, 'tinytc_recipe_tall_and_skinny_set_args', c_void_p,
         c_uint32, c_size_t, c_void_p, TinyTCMem, c_uint32, TinyTCMem,
         c_uint32, c_size_t, c_void_p, TinyTCMem, c_uint32),
        (c_int, 'tinytc_recipe_handler_release', c_void_p),
        (c_int, 'tinytc_recipe_release', c_void_p),
        (c_int, 'tinytc_core_info_release', c_void_p)
    ]


class OpenCLTinyTCKernels(OpenCLKernelProvider):
    def __init__(self, backend):
        super().__init__(backend)

        self.handle = c_void_p()

        # Recipe and timing caches
        self._recipes = {}
        self._mul_timing = {}

        # Load and wrap TinyTC
        self.lib = TinyTCWrappers()

        # Init
        self.lib.tinytc_cl_core_info_create(self.handle, backend.cl.dev)


    def __del__(self):
        for r in self._recipes.values():
            self.lib.tinytc_recipe_release(r)

        if self.handle:
            self.lib.tinytc_core_info_release(self.handle)

    def _get_tall_and_skinny_recipe(self, stype, n, k):
        try:
            return self._recipes[stype, n, k]
        except KeyError:
            recipe = c_void_p()
            self.lib.tinytc_recipe_tall_and_skinny_create(recipe, self.handle,
                                                          stype, n, k, 0, None)

            self._recipes[stype, n, k] = recipe

            return recipe

    def mul(self, a, b, out, alpha=1.0, beta=0.0):
        cl = self.backend.cl
        w, h = self.lib, self.handle

        # Ensure the matrices are compatible
        if a.nrow != out.nrow or a.ncol != b.nrow or b.ncol != out.ncol:
            raise ValueError('Incompatible matrices for out = a*b')

        m, n, k = b.ncol, a.nrow, a.ncol
        A, B, C = b, a, out

        if a.dtype == np.float64:
            alpha_ct, beta_ct = c_double(alpha), c_double(beta)
            csize = 8
            stype = w.SCALAR_TYPE_F64
        else:
            alpha_ct, beta_ct = c_float(alpha), c_float(beta)
            csize = 4
            stype = w.SCALAR_TYPE_F32

        # Create a tall-and-skinny recipe
        recipe = self._get_tall_and_skinny_recipe(stype, n, k)

        # Cache key
        ckey = (a.dtype, alpha, beta, m, n, k, a.leaddim, b.leaddim,
                out.leaddim)

        # Obtain pointers to the underlying cl_mem buffers
        ABC_p = [c_void_p(x._as_parameter_) for x in [A, B, C]]
        ABC_t = [TinyTCMem(addressof(p), w.MEM_TYPE_BUF) for p in ABC_p]
        A_t, B_t, C_t = ABC_t

        # Create the associated handler
        handler = c_void_p()
        w.tinytc_cl_recipe_handler_create(handler, cl.ctx, cl.dev, recipe)

        try:
            # Set the arguments
            w.tinytc_recipe_tall_and_skinny_set_args(
                handler, m, csize, byref(alpha_ct), A_t, A.leaddim, B_t,
                B.leaddim, csize, byref(beta_ct), C_t, C.leaddim
            )

            # Obtain the performance of the kernel
            try:
                dt = self._mul_timing[ckey]
            except KeyError:
                # Save a copy of the contents of the output matrix
                out_np = getattr(out, 'parent', out).get()

                def gemm(queue):
                    evt_ptr = c_void_p()

                    w.tinytc_cl_recipe_handler_submit(handler, queue, 0, None,
                                                      evt_ptr)

                    return cl.event(evt_ptr)

                # Benchmark the kernel and update the cache
                self._mul_timing[ckey] = dt = self._benchmark(gemm)

                # Restore the output matrix
                getattr(out, 'parent', out).set(out_np)
        except:
            w.tinytc_recipe_handler_release(handler)
            raise

        class MulKernel(OpenCLKernel):
            def __del__(self):
                w.tinytc_recipe_handler_release(handler)

            def run(self, queue, wait_for=None, ret_evt=False):
                evt_ptr = c_void_p() if ret_evt else None

                if wait_for:
                    nwait = len(wait_for)
                    wait_e = (c_void_p * nwait)(*[int(e) for e in wait_for])
                else:
                    nwait = 0
                    wait_e = None

                w.tinytc_cl_recipe_handler_submit(handler, queue, nwait,
                                                  wait_e, evt_ptr)

                if ret_evt:
                    return cl.event(evt_ptr)

        return MulKernel(mats=[a, b, out], dt=dt)
