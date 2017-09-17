# Tensorflow LSTM experiment

To run the training, download a zip-file with texts and index files from [runeberg.org](http://runeberg.org/).

```
python experiment.py --data_file nilsholg-txt.zip --save_path ./model1/ --train
```
And to auto complete a text:

```
python experiment.py --data_file nilsholg-txt.zip --save_path ./model1/ --text="Det var en " --num_words=20
```

Note that currently only words that are in the training data are supported for auto complete.
