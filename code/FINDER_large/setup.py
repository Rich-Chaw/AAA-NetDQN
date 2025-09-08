from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [
                    Extension('PrepareBatchGraph', sources = ['./FINDER_large/PrepareBatchGraph.pyx','./FINDER_large/src/lib/PrepareBatchGraph.cpp','./FINDER_large/src/lib/graph.cpp','./FINDER_large/src/lib/graph_struct.cpp',  './FINDER_large/src/lib/disjoint_set.cpp'],language='c++',extra_compile_args=['-std=c++11']),
                   Extension('graph', sources=['./FINDER_large/graph.pyx', './FINDER_large/src/lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('mvc_env', sources=['./FINDER_large/mvc_env.pyx', './FINDER_large/src/lib/mvc_env.cpp', './FINDER_large/src/lib/graph.cpp','./FINDER_large/src/lib/graph_utils.cpp','./FINDER_large/src/lib/disjoint_set.cpp', './FINDER_large/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('utils', sources=['./FINDER_large/utils.pyx', './FINDER_large/src/lib/utils.cpp', './FINDER_large/src/lib/graph.cpp', './FINDER_large/src/lib/graph_utils.cpp', './FINDER_large/src/lib/disjoint_set.cpp', './FINDER_large/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem', sources=['./FINDER_large/nstep_replay_mem.pyx', './FINDER_large/src/lib/nstep_replay_mem.cpp', './FINDER_large/src/lib/graph.cpp', './FINDER_large/src/lib/mvc_env.cpp','./FINDER_large/src/lib/graph_utils.cpp', './FINDER_large/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem_prioritized',sources=['./FINDER_large/nstep_replay_mem_prioritized.pyx', './FINDER_large/src/lib/nstep_replay_mem_prioritized.cpp','./FINDER_large/src/lib/graph.cpp','./FINDER_large/src/lib/mvc_env.cpp','./FINDER_large/src/lib/graph_utils.cpp', './FINDER_large/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['./FINDER_large/graph_struct.pyx', './FINDER_large/src/lib/graph_struct.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('GraphDQN', sources = ['./FINDER_large/GraphDQN.pyx'])
                   ])