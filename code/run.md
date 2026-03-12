python -u "./FINDER_percolation/setup.py" build_ext -i
python -u "./FINDER_percolation/testSynthetic.py"
python -u "./FINDER_percolation/testReal.py"

import tensorflow as tf 
print(tf.__version__)
print(tf.test.is_gpu_available())

cd code

CUDA_VISIBLE_DEVICES=gpu_id 

python -u "./FINDER/setup.py" build_ext -i
python -u "./FINDER/train.py"
python -u "./FINDER/testSynthetic.py"


python -u "./testReal.py"

