# ReFocus_TIR_Tracking  
The official implementation for the **\[TNNLS 2024\]** paper: "**Refocus the Attention for Parameter-Efficient Thermal Infrared Object Tracking**". 

:rocket: Update Models and Results (2024/07/17)  
[Models & Raw Results & Training logs](https://drive.google.com/drive/folders/1Q3NYwKwm-NBPKLxxApRHxNfyCgTbkOy2?usp=drive_link) [Google Driver]  
[Models & Raw Results & Training logs](https://pan.baidu.com/s/1beeHhL9wj7R8TMg3VdGvbg ) [Baidu Driver: v27s]


## Install the environment

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
prepare the LSOTB dataset. It should look like:
   ```
   ${PROJECT_ROOT}
    -- LSOTB
        -- Train
            |-- TrainingData
            |-- MyAnnotations
            ...
        -- Eval
            |-- aircraft_car
            |-- airplane_H_001
            |-- LSOTB-TIR-120.json
            |-- LSOTB-TIR-136.json
            |-- LSOTB-TIR-LT11.json
            |-- LSOTB-TIR-ST100.json
   ```

## Training
Download pre-trained `OSTrack_ep0300.pth.tar` from above driver link and put it under `$PROJECT_ROOT$/pretrained_models` . Then  

```
bash xtrain.sh
```

Replace `--config` with the desired model config under `experiments/refocus`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Evaluation
Download the model weights `OSTrack_ep0060.pth.tar` from above driver link  

Put the downloaded weights on `$PROJECT_ROOT$/checkpoints/`  

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths. Then  

```
bash ytest.sh
```

## Acknowledgments
- This repo is based on [OSTrack](https://github.com/botaoye/OSTrack) which is an excellent work.  

## Contact
If you have any question, feel free to email laisimiao@mail.dlut.edu.cn. ^_^
