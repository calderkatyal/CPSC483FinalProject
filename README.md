**LINUX SETUP:**
'''
conda create -n AutoHGNNVenv python=3.10

pip install torch==2.1.0

pip install torchdata==0.7.0

conda install -c dglteam/label/th21_cpu dgl

pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+$cu121.html

pip install "numpy<2"

pip install packaging

conda install pandas (pip install pandas=2.2.3)

conda install scikit-learn

pip install sortednp

git clone https://github.com/Yangxc13/sparse_tools.git --depth=1

cd sparse_tools

python setup.py develop
'''
