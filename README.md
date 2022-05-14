# Prerequisites

## Python

```
pip install -r requirements.txt
chmod 744 ./tools/*.py
python setup.py develop --user
```

## Data

Prepare imagenet data as 
```
/path/imagenet
|_ train
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ val
|  |_ n01440764
|  |_ ...
|  |_ n15075141
|_ ...
```

```
mkdir -p /path/pycls/pycls/datasets/data
ln -sv /path/imagenet /path/pycls/pycls/datasets/data/imagenet
```

## Command

Modify "results" to specify our dir.

```
./tools/run_net.py --mode train \
    --cfg configs/energy_effnet/regnetx/RegNetX-32GF_8_gpu.yaml \
    OUT_DIR results \
    LOG_DEST file
```

If using different number of gpus (other than 8), `TRAIN.BATCH_SIZE` and `OPTIM.BASE_LR` need to be modified. Compare `RegNetX-32GF.yaml` and `RegNetX-32GF_8_gpu.yaml`.
