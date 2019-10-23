# SiamDW_LT

If you failed to install and run this tracker, please email me (lzuqer@gmail.com).

## Prerequisites
### Install python packages
The python version should be 3.6.x, if not please run,
```
conda install -y python=3.6
```

and then run,
```
bash install.sh
```

### Download pretrained model
Download models [here](https://drive.google.com/open?id=1fJ_V5WCKROoBseLBQk3xALBqWaMb0kY8).
Unzip networks.zip in `SiamDW_LT/test/`

### Set pretrained model path
- row of 14th in `SiamDW_LT/test/settings/resnext_far/resnext.py`,
  please modify `main_path` to your code `SiamDW_LT/test` directory. eg.
  ```
  main_path = '/home/v-had/VOT2019/SiamDW_LT/test/'
  ```

  ### Prepare data
  You can creat a soft link in test dir. eg.
  ```
  ln -s $Your-LongTerm-data-path data
  ```
  or just move LongTerm data to `SiamDW_LT/test/data`.
  Then, modify `self.votlt19_path` in `SiamDW_LT/test/settings/envs.py` to your LongTerm data path.

  ### Prepare to run
  Define your experiments in `SiamDW_LT/test/settings/exp.py`. The default is LongTerm.

  ## Run tracker
  Set gpus id and processes in `SiamDW_LT/test/parallel_test.py`.
  Then run it.
  ```
  python parallel_test.py

