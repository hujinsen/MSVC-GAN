# MSVC-GAN Singing Voice Conversion

This is a tensorflow implementation of the paper: **Non-parallel Many-to-many Singing Voice Conversion by Adversarial Learning**



## Dependencies

- Python 3.6 (or higher)
- tensorflow 1.8
- librosa
- pyworld
- tensorboard
- scikit-learn



## Usage

The data directory now looks like this, put train dataset in `singers` and test dataset in `singers_test`

```text
MSVC-GAN/
├── convert_all.py #perfrom many-to-many convert use convert.py
├── convert.py #perform convert
├── model.py #model definition 
├── module.py #model's components
├── preprocess.py #preprocess the speech segments
├── README.md
├── run.py #preprocess,train,convert all in this file
├── train.py #train model
├── utility.py #utility functions
├── utils.py #common loss definition
├── data #dataset directory
│   ├── singers #train dataset
│   └── singers_test #test dataset

```

Then  run the following command,  it will `preprocess the data`, `train the model` and `convert the songs`:

```python
python run.py
```

## Demo webpage

This is the [Demo](https://hujinsen.github.io/) of the converted singing voice.

