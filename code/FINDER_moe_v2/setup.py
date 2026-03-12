from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [
                    Extension('GraphDQN', sources = ['./FINDER_moe_v2/GraphDQN.pyx']),
                    Extension('MoEGraphDQN', sources = ['./FINDER_moe_v2/MoEGraphDQN.pyx']),
                   ])