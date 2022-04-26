from procedures.Training_LUNGMASK import *

print("Training CT-GAN Injector...")


CTGAN_inj = Trainer(isInjector = False, healthy_samples=configcovid_LUNGMASK['healthy_samples_test_array'])
number_epochs=100

CTGAN_inj.train(epochs=number_epochs, batch_size=32, sample_interval=50)
print('Done.')
