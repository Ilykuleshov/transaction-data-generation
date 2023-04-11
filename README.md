# Out-Of-Distribution Detection for Event Sequences

**Student:** Belousov Nikita

**Scientific advisor:** Zaytsev Alexey

## Problem

In this project we consider a challenging problem of anomaly detection algorithms. We will concentrate on distinguishing abnormal financial data for each unique user.

Our main approach will be to use autoencoding techniques for data reconstruction. As a result we hope to get abnormal loss growth for the anomaly samples.

## Requirements

All the requirements are listed in `requirements.txt`

For install all packages run.

```
pip install -r requirements.txt
```

## Configs

| Config           | Location                                    |
| ---------------- | ------------------------------------------- |
| Main config      | [config.yaml](/config/config.yaml)          |
| Datasets         | [datasets](/config/dataset/)                |
| Embedding models | [embed_model](/config/embed_model/)         |
| Models           | [model](/config/model/)(UNDER CONSTRUCTION) |

IMPORTANT!
For now repo is on construction. Config launch is available only for tr2vec feature.

## Data

You can find all of the necessary data in [here](https://disk.yandex.ru/d/pYijj1fHonHRSw).

Now only one dataset is available (`new_data`). To begin experiments you need to place `transactions.parquet` file into `data/new_data` directory.

## Experiments

To launch experiments simply run the following command:

```
python main.py
```

with necessary config parameters.

Logs, model and results you can find on my commet (tags with `diploma` suffix) [here](https://www.comet.com/nokiroki1#projects).

## Results

As the result, we trained two autoencoder models and meta-classifier for distinguishing anomaly transactions. Results we've got tell us about possibility of out method. So, as our future work with this project we will be moving rapidly toward the GAN methods.


