This is the repository of CTGAN, a collection of Deep Learning based synthetic data generators for single table data, which are able to learn from real data and generate synthetic data with high fidelity.

On top of CTGAN, we added cyclical data transformation for date features.

For more information please see https://github.com/sdv-dev/CTGAN.


Currently, this library implements the **CTGAN** and **TVAE** models described in the [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503) paper, presented at the 2019 NeurIPS conference.

# Install

## Use CTGAN through the SDV library

:warning: If you're just getting started with synthetic data, we recommend installing the SDV library which provides user-friendly APIs for accessing CTGAN. :warning:

The SDV library provides wrappers for preprocessing your data as well as additional usability features like constraints. See the [SDV documentation](https://bit.ly/sdv-docs) to get started.

## Use the CTGAN standalone library

Alternatively, you can also install and use **CTGAN** directly, as a standalone library:

**Using `pip`:**

```bash
pip install ctgan
```

**Using `conda`:**

```bash
conda install -c pytorch -c conda-forge ctgan
```

When using the CTGAN library directly, you may need to manually preprocess your data into the correct format, for example:

* Continuous data must be represented as floats
* Discrete data must be represented as ints or strings
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

ctgan = CTGAN(epochs=10)
ctgan.fit(real_data, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(1000)
```

*For more information about the dataset see:
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.
