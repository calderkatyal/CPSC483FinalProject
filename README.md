**Overview**

Main repository for "Attention-Driven Metapath Encoding in Heterogeneous Graphs." The project uses the Pytorch Geometric framework and implements two attention-based metapath instance encoders. It furthermore implements from scratch the Loss-Aware Training Scheduler outlined in "Loss-aware Curriculum Learning for Heterogeneous Graph Neural Networks" [https://arxiv.org/abs/2402.18875](url). Included is an implementation of the base model HAN [https://arxiv.org/abs/1903.07293](url) which has been migrated to PyTorch Geometric and optimized. The model is trained on the IMDB dataset released in [https://arxiv.org/abs/2112.14936](url) using the CrawlScript found at [https://github.com/CrawlScript/gnn_datasets/](url).

**Setup:**
```
pip install -r requirements.txt

DOWNLOADS_DIR="./downloads"
DATASETS_DIR="./data"

mkdir $DOWNLOADS_DIR 
mkdir $DATASETS_DIR
wget -P  https://github.com/CrawlScript/gnn_datasets/raw/master/HGB/IMDB.zip
unzip $DOWNLOADS_DIR/IMDB.zip -d $DATASETS_DIR
```
Note that CUDA is reccomended (simply pick any distribution with CUDA support). 

**Usage**

To run our model: 
### **Usage**

To train our model, you can use the following command:

```bash
python main.py [options]
```

#### Arguments:
- `--with_lts`: Use the Loss-Aware Training Scheduler (LTS) for training. This enables the adaptive training schedule described in the paper. Default: `False`.
- `--patience`: The number of epochs to wait for improvement before early stopping. Default: `100`.
- `--lam`: Initial proportion of nodes (denoted as λ in the paper) for LTS. Default: `0.2`.
- `--T`: The epoch at which the training schedule reaches the full dataset (LTS). Default: `50`.
- `--scheduler`: Type of scheduler to use with LTS. Options: `linear`, `root`, or `geom`. Default: `linear`.
- `--epochs`: The total number of epochs to train the model. Default: `200`.
- `--metapath_encoder`: Specify the metapath encoder type. Options: `multihop`, `direct`, `mean`. Default: `multihop`.

#### Example Usage:
1. To train the model without LTS:
   ```bash
   python main.py
   ```
2. To train the model with LTS using a geometric scheduler:
   ```bash
   python main.py --with_lts --scheduler geom --epochs 200 --lam 0.3 --T 60
   ```




To run the vanilla HAN model:
```
python han_main
```
