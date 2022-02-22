# MAIC: Multi-Agent Incentive Communication with Decentralized Teammate Modeling

This is the implementation of the paper "Multi-Agent Incentive Communication via Decentralized Teammate Modeling". This repo is currently maintained by the [LAMDA-RL](https://github.com/LAMDA-RL) group.

Note: the experiments of MAIC is conducted in SC2.4.6.2.69232, which is same as the SMAC run data release (https://github.com/oxwhirl/smac/releases/tag/v1). The results are not always comparable with results run in SC2.4.10.

## Installation instructions

Set up StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.6.2.69232 into the 3rdparty folder and copy the maps necessary to run over. You may also need to set the environment variable for SC2:

```bash
export SC2PATH=[Your SC2 folder like /abc/xyz/3rdparty/StarCraftII]
```

Install Python environment with conda:

```bash
conda create -n pymarl python=3.7 -y
conda activate pymarl
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger tensorboard tensorboardx

# install qplex smac
pip install qplex_smac/
```

or install with `requirements.txt` of pip:

```bash
pip install -r requirements.txt

# install qplex smac
pip install qplex_smac/
```

As MAIC contains implementation with QPLEX mixing network (https://github.com/wjh720/QPLEX), please install the QPLEX version of SMAC in `qplex_smac` folder. You can also run algorithms without QPLEX mixing network integration with default SMAC installed (https://github.com/oxwhirl/smac). 


## Run an experiment 

```shell
python3 src/main.py --config=[Algorithm name] --env-config=[Env name] with env_args.map_name=[Map name if choosing SC2 env]
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs` includeing MAIC and other baselines.
`--env-config` refers to the config files in `src/config/envs` including `sc2`, `foraging` as the LB-Foraging environment (https://github.com/semitable/lb-foraging), `join1` as the hallway environment (https://github.com/TonghanWang/NDQ).

All results will be stored in the `results` folder.

For example, run MAIC with QMIX mixing network (default SC2 evaluation in the paper) :

```
python src/main.py --config=maic --env-config=sc2 with env_args.map_name=MMM2 seed=42
```

Run MAIC with QPLEX mixing network:

```
python src/main.py --config=maic_qplex --env-config=sc2 with env_args.map_name=MMM2 seed=42
```
