from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [
                    Extension('PrepareBatchGraph', sources = ['./FINDER/PrepareBatchGraph.pyx','./FINDER/src/lib/PrepareBatchGraph.cpp','./FINDER/src/lib/graph.cpp','./FINDER/src/lib/graph_struct.cpp',  './FINDER/src/lib/disjoint_set.cpp'],language='c++',extra_compile_args=['-std=c++11']),
                   Extension('graph', sources=['./FINDER/graph.pyx', './FINDER/src/lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('mvc_env', sources=['./FINDER/mvc_env.pyx', './FINDER/src/lib/mvc_env.cpp', './FINDER/src/lib/graph.cpp','./FINDER/src/lib/graph_utils.cpp','./FINDER/src/lib/disjoint_set.cpp', './FINDER/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('utils', sources=['./FINDER/utils.pyx', './FINDER/src/lib/utils.cpp', './FINDER/src/lib/graph.cpp', './FINDER/src/lib/graph_utils.cpp', './FINDER/src/lib/disjoint_set.cpp', './FINDER/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem', sources=['./FINDER/nstep_replay_mem.pyx', './FINDER/src/lib/nstep_replay_mem.cpp', './FINDER/src/lib/graph.cpp', './FINDER/src/lib/mvc_env.cpp','./FINDER/src/lib/graph_utils.cpp', './FINDER/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem_prioritized',sources=['./FINDER/nstep_replay_mem_prioritized.pyx', './FINDER/src/lib/nstep_replay_mem_prioritized.cpp','./FINDER/src/lib/graph.cpp','./FINDER/src/lib/mvc_env.cpp','./FINDER/src/lib/graph_utils.cpp', './FINDER/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['./FINDER/graph_struct.pyx', './FINDER/src/lib/graph_struct.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('GraphDQN', sources = ['./FINDER/GraphDQN.pyx'])
                   ])