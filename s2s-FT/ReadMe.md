**简介**

下面的是训练的命令行的参照方式，需要根据实际情况进行修改

**LSTM+LSTM**

~~~
python train_model.py -encoder LSTM -decoder LSTM -train_dataset ./test_seq_100_train.csv -test_dataset ./test_seq_100_test.csv -cuda 0
~~~

**Transformer+LSTM**

~~~
python train_model.py -encoder Transformer -decoder LSTM -train_dataset ./test_seq_100_train.csv -test_dataset ./test_seq_100_test.csv -cuda 0
~~~

**LSTM+Transformer**

~~~
python train_model.py -encoder LSTM -decoder Transformer -train_dataset ./test_seq_100_train.csv -test_dataset ./test_seq_100_test.csv -cuda 0
~~~

**Trnasformer+LSTM**

~~~
python train_model.py -encoder Transformer -decoder Transformer -train_dataset ./test_seq_100_train.csv -test_dataset ./test_seq_100_test.csv -cuda 0
~~~

