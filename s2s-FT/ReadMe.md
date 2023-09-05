**LSTM+LSTM**

~~~
python train_model.py -encoder LSTM -decoder LSTM -da
taset ./data_920.csv
~~~

**Transformer+LSTM**

~~~
python train_model.py -encoder Transformer -decoder LSTM -da
taset ./data_920.csv
~~~

**LSTM+Transformer**

~~~
python train_model.py -encoder LSTM -decoder Transformer -da
taset ./data_920.csv
~~~

**Trnasformer+LSTM**

~~~
python train_model.py -encoder Transformer -decoder Transformer -da
taset ./data_920.csv
~~~

