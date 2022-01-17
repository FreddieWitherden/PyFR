# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base import (BaseKernelProvider,
                                BasePointwiseKernelProvider, Kernel,
                                MetaKernel)
from pyfr.backends.opencl.generator import OpenCLKernelGenerator
from pyfr.nputil import npdtype_to_ctypestype
from pyfr.util import memoize


class OpenCLKernel(Kernel):
    def add_to_graph(self, graph, deps):
        pass


class _OpenCLMetaKernelCommon(object):
    def add_to_graph(self, graph, deps):
        pass


class OpenCLOrderedMetaKernel(_OpenCLMetaKernelCommon, MetaKernel):
    def run(self, queue, wait_for=None, ret_evt=False):
        for k in self.kernels[:-1]:
            wait_for = [k.run(queue, wait_for, True)]

        return self.kernels[-1].run(queue, wait_for, ret_evt)


class OpenCLUnorderedMetaKernel(_OpenCLMetaKernelCommon, MetaKernel):
    def run(self, queue, wait_for=None, ret_evt=False):
        if ret_evt:
            kevts = [k.run(queue, wait_for, True) for k in self.kernels]
            return queue.marker(kevts)
        else:
            for k in self.kernels:
                k.run(queue, wait_for, False)


class OpenCLKernelProvider(BaseKernelProvider):
    @memoize
    def _build_program(self, src):
        return self.backend.cl.program(src, flags=['-cl-fast-relaxed-math'])

    def _build_kernel(self, name, src, argtypes):
        argtypes = [npdtype_to_ctypestype(arg) for arg in argtypes]

        return self._build_program(src).get_kernel(name, argtypes)


class OpenCLPointwiseKernelProvider(OpenCLKernelProvider,
                                    BasePointwiseKernelProvider):
    kernel_generator_cls = OpenCLKernelGenerator

    def _instantiate_kernel(self, dims, fun, arglst, argmv):
        rtargs = []

        # Determine the work group sizes
        if len(dims) == 1:
            ls = (64,)
            gs = (dims[0] - dims[0] % -ls[0],)
        else:
            ls = (64, 4)
            gs = (dims[1] - dims[1] % -ls[0], ls[1])

        fun.set_dims(gs, ls)

        # Process the arguments
        for i, k in enumerate(arglst):
            if isinstance(k, str):
                rtargs.append((i, k))
            else:
                fun.set_arg(i, k)

        class PointwiseKernel(OpenCLKernel):
            if rtargs:
                def bind(self, **kwargs):
                    for i, k in rtargs:
                        fun.set_arg(i, kwargs[k])

            def run(self, queue, wait_for=None, ret_evt=False):
                return fun.exec_async(queue, wait_for, ret_evt)

        return PointwiseKernel(*argmv)
