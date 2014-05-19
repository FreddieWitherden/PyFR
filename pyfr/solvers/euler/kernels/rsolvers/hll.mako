# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

<%pyfr:macro name='rsolve' params='ul, ur, n, nf'>
    // Compute the left and right fluxes + velocities and pressures
    fpdtype_t fl[${ndims}][${nvars}], fr[${ndims}][${nvars}];
    fpdtype_t vl[${ndims}], vr[${ndims}];
    fpdtype_t pl, pr;
    fpdtype_t nf_sub[${nvars}], nf_fl[${nvars}], nf_fr[${nvars}];

    ${pyfr.expand('inviscid_flux', 'ul', 'fl', 'pl', 'vl')};
    ${pyfr.expand('inviscid_flux', 'ur', 'fr', 'pr', 'vr')};

    // Roe average velocity and take normal
    fpdtype_t nv = ${pyfr.dot('n[{i}]',
                     'sqrt(ul[0])*vl[{i}] + sqrt(ur[0])*vr[{i}]', i=ndims)}
                     / (sqrt(ul[0]) + sqrt(ur[0]));
    
    // Roe average enthalpy
    fpdtype_t H = ((ul[${nvars - 1}] + pl)/sqrt(ul[0])
                     + (ur[${nvars - 1}] + pr)/sqrt(ur[0]))
                     / sqrt(ul[0] + ur[0]);

    // Roe average sound speed
    fpdtype_t a = sqrt(${c['gamma']-1}*(H - 0.5*nv*nv));

    // Estimate the left and right wave speed, sl and sr
    fpdtype_t sl = nv - a;
    fpdtype_t sr = nv + a;

    // Tests if supersonic to the right, to the left or subsonic
    // Coefficients for summation of possible flux combinations
    fpdtype_t sign_sl = (1.0 + copysign(1.0, sl))/2.0;
    fpdtype_t sign_sr = (1.0 - copysign(1.0, sr))/2.0;
    fpdtype_t sign_other = (1-sign_sl)*(1-sign_sr);

    // Output
    // Possible solutions for flux
% for i in range(nvars):
    nf_sub[${i}] = (${' + '.join('n[{j}]*(sr*fl[{j}][{i}] - sl*fr[{j}][{i}])'
                                 .format(i=i, j=j) for j in range(ndims))}
                            + sl*sr*(ur[${i}] - ul[${i}]))/(sr-sl);
    nf_fl[${i}] = (${' + '.join('n[{j}]*fl[{j}][{i}]'
                                 .format(i=i, j=j) for j in range(ndims))});
    nf_fr[${i}] = (${' + '.join('n[{j}]*fr[{j}][{i}]'
                                 .format(i=i, j=j) for j in range(ndims))});
    nf[${i}] = sign_sl*nf_fl[${i}] + sign_sr*nf_fr[${i}]
              + sign_other*nf_sub[${i}];
% endfor

</%pyfr:macro>