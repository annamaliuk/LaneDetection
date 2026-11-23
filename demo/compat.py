import numpy as np
import numpy.random._pickle as pickle_ctors


class IgnoreState(np.random.PCG64):
	def __setstate__(self, state):
		pass


def bit_generator(name='PCG64'):
	if isinstance(name, np.random.BitGenerator):
		return name
	return IgnoreState()


pickle_ctors.__bit_generator_ctor = bit_generator
pickle_ctors.__generator_ctor = lambda name='PCG64', ctor=None: np.random.Generator(bit_generator(name))
pickle_ctors.__randomstate_ctor = lambda name='MT19937', ctor=None: np.random.RandomState(bit_generator(name))