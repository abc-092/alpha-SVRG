#  &alpha;-SVRG
Official PyTorch implementation for $\alpha$-SVRG

## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Training

### Basic Recipe
We list commands for $\alpha$-SVRG on `convnext_femto` and `vit_base` with coefficient `0.75`.
- For training other models, change `--model` accordingly, e.g., to `vit_tiny`, `convnext_base`, `vit_base`.
- For using different coefficients, change `--coefficient` accordingly, e.g., to `1`, `0.5`.
- `--use_cache_svrg` can be enabled on smaller models provided with sufficient memory and disabled on larger models.
- Our results of smaller models on ImageNet-1K were produced with 4 nodes, each with 8 gpus. Our results of smaller models on ImageNet-1K were produced with 8 nodes, each with 8 gpus. Our results of ConvNeXt-Femto on small datasets were produced with 8 gpus. 

Below we give example commands for both smaller models and larger models on ImageNet-1K and ConvNeXt-Femto on small datasets.

**Smaller models**

```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_femto --epochs 300 \
--batch_size 128 --lr 4e-3 \
--use_svrg true --coefficient 0.75 --svrg_schedule linear --use_cache_svrg true \
--data_path /path/to/data/ --data_set IMNET \
--output_dir /path/to/results/
```

**Larger models**

```
python run_with_submitit.py --nodes 8 --ngpus 8 \
--model vit_base --epochs 300 \
--batch_size 64 --lr 4e-3 \
--use_svrg true --coefficient 0.75 --svrg_schedule linear \
--data_path /path/to/data/ --data_set IMNET \
--output_dir /path/to/results/
```

**ConvNeXt-Femto on small datasets**
- Fill in `epochs`, `warmup_epochs`, and `batch_size` based on `data_set`.
- Note that `batch_size` is the batch size for each gpu.
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_femto --epochs $epochs --warmup_epochs $warmup_epochs \
--batch_size $batch_size --lr 4e-3 \
--use_svrg true --coefficient 0.75 --svrg_schedule linear --use_cache_svrg true \
--data_path /path/to/data/ --data_set $data_set \
--output_dir /path/to/results/
```
### Evaluation

single-GPU
```
python main.py --model convnext_femto --eval true \
--resume /path/to/model \
--data_path /path/to/data
```

multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_femto --eval true \
--resume /path/to/model \
--data_path /path/to/data
```


## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) codebase.