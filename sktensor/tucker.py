# sktensor.tucker - Algorithms to compute Tucker decompositions
# Copyright (C) 2013 Maximilian Nickel <mnick@mit.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.	If not, see <http://www.gnu.org/licenses/>.

import logging
import time
import numpy as np
from numpy import array, ones, sqrt
from numpy.random import rand
from .pyutils import is_number
from .core import ttm, nvecs, norm, ttv
from sktensor import dtensor
import itertools
from multiprocessing import Pool
import sys

# exports
__all__ = [
	'hooi',
	'hosvd',
]

# default values
_log = logging.getLogger('TUCKER')
_DEF_MAXITER = 500
_DEF_INIT = 'nvecs'
_DEF_CONV = 1e-7
_DEF_EDIMS = []
_DEF_CPUS = 4


def hooi(X, rank, **kwargs):
	"""
	Compute Tucker decomposition of a tensor using Higher-Order Orthogonal
	Iterations.

	Parameters
	----------
	X : tensor_mixin
		The tensor to be decomposed
	rank : array_like
		The rank of the decomposition for each mode of the tensor.
		The length of ``rank`` must match the number of modes of ``X``.
	init : {'random', 'nvecs'}, optional
		The initialization method to use.
			- random : Factor matrices are initialized randomly.
			- nvecs : Factor matrices are initialzed via HOSVD.
		default : 'nvecs'
	edims : array_like
		List of dimensions that should be treated in a memory-efficient manner.
		default : []
		Note : support only for 3-way tensors with len(edims)==1

	Examples
	--------
	Create dense tensor

	>>> from sktensor import dtensor
	>>> T = np.zeros((3, 4, 2))
	>>> T[:, :, 0] = [[ 1,	4,	7, 10], [ 2,  5,  8, 11], [3,  6,  9, 12]]
	>>> T[:, :, 1] = [[13, 16, 19, 22], [14, 17, 20, 23], [15, 18, 21, 24]]
	>>> T = dtensor(T)

	Compute Tucker decomposition of ``T`` with n-rank [2, 3, 1] via higher-order
	orthogonal iterations

	>>> Y = hooi(T, [2, 3, 1], init='nvecs')

	Shape of the core tensor matches n-rank of the decomposition.

	>>> Y['core'].shape
	(2, 3, 1)
	>>> Y['U'][1].shape
	(3, 2)

	References
	----------
	.. [1] L. De Lathauwer, B. De Moor, J. Vandewalle: On the best rank-1 and
		   rank-(R_1, R_2, \ldots, R_N) approximation of higher order tensors;
		   IEEE Trans. Signal Process. 49 (2001), pp. 2262-2271
	"""
	# init options
	ainit = kwargs.pop('init', _DEF_INIT)
	maxIter = kwargs.pop('maxIter', _DEF_MAXITER)
	conv = kwargs.pop('conv', _DEF_CONV)
	dtype = kwargs.pop('dtype', X.dtype)
	Edims = kwargs.pop('edims', _DEF_EDIMS)
	cpus = kwargs.pop('cpus', _DEF_CPUS)
	if not len(kwargs) == 0:
		raise ValueError('Unknown keywords (%s)' % (kwargs.keys()))

	ndims = X.ndim
	if is_number(rank):
		rank = rank * ones(ndims)
	
	Sdims = sorted(set(range(ndims)) - set(Edims))

	normX = norm(X)

	U = __init(ainit, X, ndims, rank, dtype)
	fit = 0
	exectimes = []
	R = [range(r) for r in rank]
	for itr in range(maxIter):
		tic = time.clock()
		fitold = fit
		
		for n in range(ndims):
			shp = [r for r in rank]
			shp[n] = X.shape[n]
			Utilde = dtensor(np.zeros(shp))
			xedims = [(rank[e],e) for e in range(ndims) if e!=n]
			xedims.sort(reverse=True)
			if len(Edims) > 0:
				edims = [xedims[-1][1]]
				sdims = [s[1] for s in xedims[:-1]]
			else:
				edims = []
				sdims = [s[1] for s in xedims]
			Rn = [R[s] for s in edims]
			Nr = len(list(itertools.product(*R)))
			for args in jlist_generator(Rn,U,edims,sdims,X,n):
				i,tube = Utilde_tube(args)
				Utilde[i] = tube
			U[n] = nvecs(Utilde, n, rank[n])
			print 'edims calculated for iter %d, mode %d' % (itr,n)
			sys.stdout.flush()

		# compute core tensor to get fit
		core = ttm(Utilde, U, n, transp=True)

		# since factors are orthonormal, compute fit on core tensor
		normresidual = sqrt(normX ** 2 - norm(core) ** 2)

		# fraction explained by model
		fit = 1 - (normresidual / normX)
		fitchange = abs(fitold - fit)
		exectimes.append(time.clock() - tic)
		print '[%3d] fit: %.5f | delta: %7.1e | secs: %.5f' % (itr, fit, fitchange, exectimes[-1])
		
		_log.debug(
			'[%3d] fit: %.5f | delta: %7.1e | secs: %.5f'
			% (itr, fit, fitchange, exectimes[-1])
		)
		if itr > 1 and fitchange < conv:
			break
	return core, U

def jlist_generator(Rn,U,edims,sdims,X,n):
	if len(Rn) > 0:
		for jlist in itertools.product(*Rn):
			yield (jlist,[U[edims[i]][:,j] for i,j in enumerate(jlist)] + [U[s] for s in sdims],X,edims + sdims,n)
	else:
		yield ([],[U[s] for s in sdims],X,sdims,n)

def Utilde_tube(args):
	jlist,U,X,dims,n = args
	ttv_dims = []
	ut,td = ttm_or_ttv(X,U[0],dims[0])
	if td:
		ttv_dims.append(dims[0])
	for i in range(1,len(dims)):
		mode = dims[i] - sum(d < dims[i] for d in ttv_dims)
		print 'jlist:',jlist,'ut shape:',ut.shape,'U[i] shape:',U[i].shape,mode
		sys.stdout.flush()
		ut,td = ttm_or_ttv(ut,U[i],mode)
		if td:
			ttv_dims.append(dims[i])
	idx = [_ for _ in X.shape]
	idx[n] = slice(0,X.shape[n])
	for i,d in enumerate(dims):
		if d in ttv_dims:
			idx[d] = jlist[ttv_dims.index(d)]
		else:
			idx[d] = slice(0,U[i].shape[1])
	if not isinstance(ut,np.ndarray):
		ut = ut.toarray()
	return (idx,ut)

def ttm_or_ttv(x,v,mode):
	if len(v.shape) == 1:
		return x.ttv(v,modes=[mode]),True
	else:
		return x.ttm(v,mode,transp=True),False


def hosvd(X, rank, dims=None, dtype=None, compute_core=True):
	U = [None for _ in range(X.ndim)]
	if dims is None:
		dims = range(X.ndim)
	if dtype is None:
		dtype = X.dtype
	for d in dims:
		U[d] = array(nvecs(X, d, rank[d]), dtype=dtype)
	if compute_core:
		core = X.ttm(U, transp=True)
		return U, core
	else:
		return U


def __init(init, X, N, rank, dtype):
	# Don't compute initial factor for first index, gets computed in
	# first iteration
	Uinit = [None]
	if isinstance(init, list):
		Uinit = init
	elif init == 'random':
		for n in range(1, N):
			Uinit.append(array(rand(X.shape[n], rank[n]), dtype=dtype))
	elif init == 'nvecs':
		Uinit = hosvd(X, rank, range(1, N), dtype=dtype, compute_core=False)
	return Uinit


