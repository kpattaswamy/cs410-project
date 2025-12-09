# cs410-project

## Setup datasets locally

1. `mkdir datasets`
2. `cd datasets/`
3. Download datasets from: `https://www.kaggle.com/datasets/injek0626/reddit-stock-related-posts?resource=download`

## Install requirements

`pip3 install -r requirements.txt`

## Usage

### Preprocess datasets
```
python3 dataset-gen.py
```
Creates `posts_with_symbols.csv` which contains post data along with the ticker symbol. This information is sourced from the files in `datasets/`

```
python3 label.py
```
Creates the file: `preprocessed-dataset/labeled.csv`. Adds a binary label (indicating if price moved up or down the following trading day) to the data from `posts_with_symbols.csv`. The script queries the `yfinance` API to fetch market data about price movement.

### Train the model
```
python3 model.py
```
Trains the model locally using `preprocessed-dataset/labeled.csv` and saves at `./bert_stock_predictor_final`.

### Run the model locally to predict price movement
```
python3 predict.py
```
Stands up the trained model locally and takes input of a ticker symbol and text. It then predicts if price will move up or down.

### Results while testing
* Epoch 1/3, Average Loss: 0.6947
* Epoch 2/3, Average Loss: 0.6596
* Epoch 3/3, Average Loss: 0.5863

* Test Accuracy: 0.5696
* Test F1: 0.5401