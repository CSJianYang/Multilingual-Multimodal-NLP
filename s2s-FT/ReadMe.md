**简介**
下面的是训练的命令行的参照方式，需要根据实际情况进行修改

**LSTM+LSTM**

~~~
python train_model.py -encoder LSTM -decoder LSTM -dataset ./data_920_new.csv -cuda 3
~~~

**Transformer+LSTM**

~~~
python train_model.py -encoder Transformer -decoder LSTM -dataset ./data_920_new.csv -cuda 3
~~~

**LSTM+Transformer**

~~~
python train_model.py -encoder LSTM -decoder Transformer -dataset ./data_920_new.csv -cuda 2
~~~

**Trnasformer+LSTM**

~~~
python train_model.py -encoder Transformer -decoder Transformer -dataset ./data_920_new.csv -cuda 1
~~~

