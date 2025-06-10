# Semcor Experiments
This folder contains the data and the code used to run the experiments of our study.

### 🔬 To reproduce the experiments on **Semcor**:

1. Create a conda environment

```
conda create -n env-name python==3.9
```

2. Install the requirements

```
pip install -r requirements.txt
```

3. Launch the experiments

```
python3 main.py k-value
```

📊 Results of all the experiments made with a model will be saved in `./errors/K-value/model-name/...`
