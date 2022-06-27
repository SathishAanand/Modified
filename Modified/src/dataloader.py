from dataclasses import dataclass
import os
import glob

import tensorflow as tf

#from network import NetworkConfiguration, getDataset
from network import *
from trainer import *
from configuration import *


@dataclass
class Datasplits:
    training_data: tf.float16
    training_labels: tf.float16
    testing_data: tf.float32
    testing_labels: tf.float32


class Dataloader:
    def __init__(self, config: NetworkConfiguration) -> None:
        self.config = config

        test_data_path = self.config.test_dataset_path
        train_data_path = self.config.train_dataset_path

        self.test_data = glob.glob(os.path.join(test_data_path, "*run100*State*1.400[0-2]0*.h5"))
        self.train_data = glob.glob(os.path.join(train_data_path, "*run10[1-4]*State*1.[0-4]00[0-2]0*.h5"))

    def get_dataset_and_labels(self):
        train_dataset, train_labels, test_dataset, test_labels = getDataset(self.train_data,
                                                                            self.test_data,
                                                                            self.config.dt,
                                                                            self.config.sequence_lenght,
                                                                            debug=self.config.debug_level,
                                                                            doSqueeze=True,
                                                                            doShuffle=True,
                                                                            doDataAug=True)

        return Datasplits(training_data=train_dataset,
                          training_labels=train_labels,
                          testing_data=test_dataset,
                          testing_labels=test_labels)
