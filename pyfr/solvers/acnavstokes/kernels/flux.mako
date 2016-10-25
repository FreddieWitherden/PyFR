# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='viscous_flux_add' params='uin, grad_uin, fout'>

% for i in range(ndims):
    fout[${i}][1] += -${c['nu']}*grad_uin[${i}][1];
% endfor

% for i in range(ndims):
    fout[${i}][2] += -${c['nu']}*grad_uin[${i}][2];
% endfor

% if ndims == 3:
% for i in range(ndims):
    fout[${i}][3] += -${c['nu']}*grad_uin[${i}][3];
% endfor
% endif
</%pyfr:macro>