# GAN
This is a Multi Purpose GAN (MP-GAN) playground I built to experiment with architectures and datasets.\
I tested it both on images (mnist) and tabular data. To switch between tabular data and images you only need to change the\
model structure, not the internal flow.\
Usage:\
The training flow and internal behavior is managed by the files in ./models folder.\
The User configures the model structure and training parameters, as well as the training data, from the main notebook - GPGAN_Notebook.ipynb.\
If you want to add support for more layers - add them to ./models/deep_model.py.\
In gan_utils.py I implemented simple functions to estimate the utility of synthesized tabular data: While in images, it's easy for humans to\
tell whether the synthetic output is valid, when it comes to tabular data (especially if it consists of many features), it's harder to do it.\
Two common examination methods look at individual feature distribution, and mutual correlations between feature. You expect the distributions\
and correlations to be approximately the same in the original and synthetic data.\

Some results:\
The image below was generated using a small GAN (D size 40k params, G size 1M params), training for ~600 epochs on MNIST.\
<img width="579" alt="image" src="https://user-images.githubusercontent.com/47942735/209480564-342538c2-2426-4d82-b189-b88e9702e8ae.png">
Next, is a comparison between 2 larger GANs (same G, but D 160k params), after 10, 20 and 50 epochs of training. Left: train from scratch,\
right: use predtrained D.\
<img width="718" alt="image" src="https://user-images.githubusercontent.com/47942735/209481169-36867ac0-6db0-4a00-bbfd-ad7702b43c75.png">
<img width="714" alt="image" src="https://user-images.githubusercontent.com/47942735/209481107-b79548fb-f125-492a-9d1a-8c741a6b8c85.png">
