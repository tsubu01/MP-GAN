# GAN
This is a General Purpose GAN (GPGAN) playground where I can experiment with architectures and datasets.
I tested it both on images (mnist) and tabular data. To switch between tabular data and images you only need to change the
model structure, not the internal flow.
Usage:
The training flow and internal behavior is managed by the files in ./models folder.
The User configures the model structure and training parameters, as well as the training data, from the main notebook - GPGAN_Notebook.ipynb.
If you want to add support for more layers - add them to ./models/deep_model.py.
In gan_utils.py I implemented simple functions to estimate the utility of synthesized tabular data: While in images, it's easy for humans to
tell whether the synthetic output is valid, when it comes to tabular data (especially if it consists of many features), it's harder to do it.
Two common examination methods look at individual feature distribution, and mutual correlations between feature. You expect the distributions
and correlations to be approximately the same in the original and synthetic data.
