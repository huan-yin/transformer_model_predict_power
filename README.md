# Transformer Model Predict Power

This Transformer model is used to predict the future total active power based on the provided data. It can predict the data of a specified number of days in the future based on the data of a specified number of days in the past.

## Dataset Preparation

Create directory ``data``, download dataset from [there](https://drive.google.com/drive/folders/1L8Y2BzK6f_reLAYo0ryW1nD4ZAou85Qh?usp=sharing), put it to the directory ``data``.


## Prepare the Runtime Environment and Required Dependencies
Using conda create python environment
```
conda create -n transformer_predict_power python=3.9
```

Activate environment

```
conda activate transformer_predict_power
```

Install the GPU version of PyTorch

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

Install the Python packages required by the project
```
pip install scikit-learn==1.6.1 pandas==2.3.1 matplotlib==3.9.4 torch-summary==1.4.5 tqdm==4.67.1 numpy==1.26.2
```

## Training
Train using the basic version of Transformer
```
python train.py
```

Train using the improved version of Transformer

```
python train.py --use_improved
```

Specifies how many days in the past and how many days in the future to train on

```
python train.py --input_days 90 --output_days 90 
```

Train with more detailed configurations
```
python train.py \
--train_data_path "data/train.csv" \
--test_data_path "data/test.csv" \
--save_dir "data/output" \
--model_name "transformer" \
--use_improved \
--batch_size 32 \
--lr 0.001 \
--epochs 500 \
--input_days 90 \
--output_days 90 \
--target_col "Global_active_power"
```