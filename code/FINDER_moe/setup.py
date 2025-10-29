from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext':build_ext},
    ext_modules = [
                    Extension('PrepareBatchGraph', sources = ['./FINDER_moe/PrepareBatchGraph.pyx','./FINDER_moe/src/lib/PrepareBatchGraph.cpp','./FINDER_moe/src/lib/graph.cpp','./FINDER_moe/src/lib/graph_struct.cpp',  './FINDER_moe/src/lib/disjoint_set.cpp'],language='c++',extra_compile_args=['-std=c++11']),
                   Extension('graph', sources=['./FINDER_moe/graph.pyx', './FINDER_moe/src/lib/graph.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('mvc_env', sources=['./FINDER_moe/mvc_env.pyx', './FINDER_moe/src/lib/mvc_env.cpp', './FINDER_moe/src/lib/graph.cpp','./FINDER_moe/src/lib/graph_utils.cpp','./FINDER_moe/src/lib/disjoint_set.cpp', './FINDER_moe/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('utils', sources=['./FINDER_moe/utils.pyx', './FINDER_moe/src/lib/utils.cpp', './FINDER_moe/src/lib/graph.cpp', './FINDER_moe/src/lib/graph_utils.cpp', './FINDER_moe/src/lib/disjoint_set.cpp', './FINDER_moe/src/lib/decrease_strategy.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem', sources=['./FINDER_moe/nstep_replay_mem.pyx', './FINDER_moe/src/lib/nstep_replay_mem.cpp', './FINDER_moe/src/lib/graph.cpp', './FINDER_moe/src/lib/mvc_env.cpp','./FINDER_moe/src/lib/graph_utils.cpp', './FINDER_moe/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('nstep_replay_mem_prioritized',sources=['./FINDER_moe/nstep_replay_mem_prioritized.pyx', './FINDER_moe/src/lib/nstep_replay_mem_prioritized.cpp','./FINDER_moe/src/lib/graph.cpp','./FINDER_moe/src/lib/mvc_env.cpp','./FINDER_moe/src/lib/graph_utils.cpp', './FINDER_moe/src/lib/disjoint_set.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('graph_struct', sources=['./FINDER_moe/graph_struct.pyx', './FINDER_moe/src/lib/graph_struct.cpp'], language='c++',extra_compile_args=['-std=c++11']),
                    Extension('GraphDQN', sources = ['./FINDER_moe/GraphDQN.pyx']),
                    Extension('MoEGraphDQN', sources = ['./FINDER_moe/MoEGraphDQN.pyx']),
                   ])