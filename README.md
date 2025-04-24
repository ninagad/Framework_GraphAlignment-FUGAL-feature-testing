# **Installation**

Create conda environment with 
```shell
conda create -n <env name>
```

Activate conda environment with
```shell
conda activate <env name>
```

Install pip with 
```shell
conda install pip
```

Install dependencies with 
```shell
pip install -r requirements.txt
```

# ** Usage **
To run with sacred named configuration use e.g. 
```shell
python workexp.py with tuning
```

The named configs can be found in the `experiment/experiments.py` file.

To run hyperparameter tuning use
```shell
python -m data_analysis.hyparam_tuning --tune <tuning option>
```
