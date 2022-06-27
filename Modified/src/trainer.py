import os
from omegaconf import OmegaConf

#from network import TurbulanceNetwork, Datasplits, Dataloader, NetworkConfiguration
from network import *
from configuration import *
from dataloader import *
from read_flexi import *


if __name__ == '__main__':

    configuration_dictionary = OmegaConf.load(os.path.abspath('/content/drive/MyDrive/Colab Notebooks/Modified/config.yaml'))
    configuration = NetworkConfiguration.from_dictionary(configuration_dictionary)

    dataloader = Dataloader(configuration)
    training_validation_data = dataloader.get_dataset_and_labels()
    turbulance_network = TurbulanceNetwork(configuration)

    turbulance_network.train_model(training_validation_data)
