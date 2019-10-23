# SiamDW-based trackers for VOT-2019 challenge

We are hiring talented interns: houwen.peng@microsoft.com

## SiamDW_D

If you failed to install and run this tracker, please email me (v-hongyy@microsoft.com).

### Prerequisites
#### Install python packages
The python version should be 3.6.x, if not please run,
```
conda install -y python=3.6
```

and then run,
```
bash install.sh
```

#### Download pretrained model
Download models [here](https://drive.google.com/file/d/1f44B3zHO9Sjz8W3IvnZRKmlHlOkBCP7i/view?usp=sharing).
Unzip networks.zip in `SiamDW_D/test/`

#### Set pretrained model path
- row of 14th in `SiamDW_D/test/settings/resnext_far/resnext.py`,
  and row of 14th in `SiamDW_D/test/settings/senet_far/senet.py`
  please modify `main_path` to your code `SiamDW_D/test` directory. eg.
```
main_path = '/home/hongyuan/VOT2019/SiamDW_D/test/'
```

#### Prepare data
You can creat a soft link in test dir. eg.
```
ln -s $Your-RGBD-data-path data 
```
or just move RGBD data to `SiamDW_D/test/data`.
Then, modify `self.rgbd_path` in `SiamDW_D/test/settings/envs.py` to your RGBD data path.

#### Prepare to run
Define your experiments in `SiamDW_D/test/settings/exp.py`. The default is RGBD.

### Run tracker
Set gpus id and processes in `SiamDW_D/test/parallel_test.py`.
Then run it.
```
python parallel_test.py
```

## SiamDW_T
If you failed run this tracker, please be free to email me (v-zhipz@microsoft.com).

### Install python packages
```
sh install.sh
```

### Download model
Download pre-trained [model](https://drive.google.com/file/d/1n2UEClYOhCFA27pz0JBshwHjx1sUYuiu/view?usp=sharing) and put it in `SiamDW_T/snapshot`

### Prepare testing data
```
cd SiamDW_T
mkdir dataset
cd dataset
ln -sfb $you_rgbt_vot_workspace/sequences VOT2019RGBT
cd ..
```


### Run
```
python rgbt_tracking/test_rgbt.py
```

## SiamDW_LT

If you failed to install and run this tracker, please email me (lzuqer@gmail.com).

### Prerequisites
#### Install python packages
The python version should be 3.6.x, if not please run,
```
conda install -y python=3.6
```

and then run,
```
bash install.sh
```

#### Download pretrained model
Download models [here](https://drive.google.com/open?id=1fJ_V5WCKROoBseLBQk3xALBqWaMb0kY8).
Unzip networks.zip in `SiamDW_LT/test/`

#### Set pretrained model path
- row of 14th in `SiamDW_LT/test/settings/resnext_far/resnext.py`,
  please modify `main_path` to your code `SiamDW_LT/test` directory. eg.
```
main_path = '/home/v-had/VOT2019/SiamDW_LT/test/'
```

#### Prepare data
You can creat a soft link in test dir. eg.
```
ln -s $Your-LongTerm-data-path data
```
or just move LongTerm data to `SiamDW_LT/test/data`.
Then, modify `self.votlt19_path` in `SiamDW_LT/test/settings/envs.py` to your LongTerm data path.

#### Prepare to run
Define your experiments in `SiamDW_LT/test/settings/exp.py`. The default is LongTerm.

### Run tracker
Set gpus id and processes in `SiamDW_LT/test/parallel_test.py`.
Then run it.
```
python parallel_test.py



