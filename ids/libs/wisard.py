# Conteúdo para o arquivo: wisard.py

import numpy as np
# Importa a classe BloomFilter do nosso outro arquivo
from libs.bloom_filter import BloomFilter

def generate_h3_values(num_inputs, num_entries, num_hashes):
    assert(np.log2(num_entries).is_integer())
    shape = (num_hashes, num_inputs)
    values = np.random.randint(0, num_entries, shape)
    return values

class Discriminator:
    def __init__(self, num_inputs, unit_inputs, unit_entries, unit_hashes, random_values=None):
        assert((num_inputs/unit_inputs).is_integer())
        self.num_filters = num_inputs // unit_inputs
        self.filters = [BloomFilter(unit_inputs, unit_entries, unit_hashes, random_values) for i in range(self.num_filters)]

    def train(self, xv):
        filter_inputs = xv.reshape(self.num_filters, -1)
        for idx, inp in enumerate(filter_inputs):
            self.filters[idx].add_member(inp)

    def predict(self, xv):
        filter_inputs = xv.reshape(self.num_filters, -1)
        response = 0
        for idx, inp in enumerate(filter_inputs):
            response += int(self.filters[idx].check_membership(inp))
        return response
    
    def set_bleaching(self, bleach):
        for f in self.filters:
            f.set_bleaching(bleach)

class WiSARD:
    def __init__(self, num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes):
        self.pad_zeros = (((num_inputs // unit_inputs) * unit_inputs) - num_inputs) % unit_inputs
        pad_inputs = num_inputs + self.pad_zeros
        self.input_order = np.arange(pad_inputs)
        np.random.shuffle(self.input_order)
        random_values = generate_h3_values(unit_inputs, unit_entries, unit_hashes)
        self.discriminators = [Discriminator(self.input_order.size, unit_inputs, unit_entries, unit_hashes, random_values) for i in range(num_classes)]

    def train(self, xv, label):
        xv = np.pad(xv, (0, self.pad_zeros), 'constant')[self.input_order]
        self.discriminators[label].train(xv)

    def predict(self, xv):
        xv = np.pad(xv, (0, self.pad_zeros), 'constant')[self.input_order]
        responses = np.array([d.predict(xv) for d in self.discriminators], dtype=int)
        max_response = responses.max()
        return np.where(responses == max_response)[0], responses

    def set_bleaching(self, bleach):
        for d in self.discriminators:
            d.set_bleaching(bleach)