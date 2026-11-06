# Conteúdo para o arquivo: bloom_filter.py

import numpy as np
from numba import jit

@jit(nopython=True)
def h3_hash(xv, m):
    selected_entries = xv * m
    reduction_result = np.zeros(m.shape[0], dtype=np.int64)
    for i in range(m.shape[1]):
        reduction_result ^= selected_entries[:,i]
    return reduction_result

class BloomFilter:
    def __init__(self, num_inputs, num_entries, num_hashes, hash_constants):
        self.num_inputs, self.num_entries, self.num_hashes = num_inputs, num_entries, num_hashes
        self.hash_values = hash_constants
        self.index_bits = int(np.log2(num_entries))
        self.data = np.zeros(num_entries, dtype=int)
        self.bleach = np.array(1, dtype=int)

    @staticmethod
    @jit(nopython=True)
    def __check_membership(xv, hash_values, bleach, data):
        hash_results = h3_hash(xv, hash_values)
        least_entry = data[hash_results].min()
        return least_entry >= bleach

    def check_membership(self, xv):
        return BloomFilter.__check_membership(xv, self.hash_values, self.bleach, self.data)
    
    @staticmethod
    @jit(nopython=True)
    def __add_member(xv, hash_values, data):
        hash_results = h3_hash(xv, hash_values)
        least_entry = data[hash_results].min()
        data[hash_results] = np.maximum(data[hash_results], least_entry+1)

    def add_member(self, xv):
        BloomFilter.__add_member(xv, self.hash_values, self.data)

    def set_bleaching(self, bleach):
        self.bleach[...] = bleach

    def binarize(self):
        self.data = (self.data >= self.bleach).astype(int)
        self.set_bleaching(1)