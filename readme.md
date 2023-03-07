This is the repository of CTGAN, a collection of Deep Learning based synthetic data generators for single table data, which are able to learn from real data and generate synthetic data with high fidelity.

On top of CTGAN, we added cyclical data transformation for date features.

For more information please see https://github.com/sdv-dev/CTGAN.


Currently, this library implements the **CTGAN** and **TVAE** models described in the [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503) paper, presented at the 2019 NeurIPS conference.


When using the CTGAN library directly, you may need to manually preprocess your data into the correct format, for example:

* Continuous data must be represented as floats
* Discrete data must be represented as ints or strings
* Date feature should be represented as 2021-03-07
* The data should not contain any missing values

# Usage Example

In this example we load the [Adult Census Dataset](https://archive.ics.uci.edu/ml/datasets/adult)* which is a built-in demo dataset. We use CTGAN to learn from the real data and then generate some synthetic data.

```python3
from ctgan import CTGAN
from ctgan import load_demo

real_data = load_demo()

# Names of the columns that are discrete
discrete_columns = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
    'income'
]

# Names of the columns that are dates
date_columns = []

ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns, date_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000, date_columns)
```

*For more information about the dataset see:
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.
