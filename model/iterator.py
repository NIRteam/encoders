import galois
import numpy as np


class MyIterator:
    """
        k:кодируемые данные
        n:последовательность с дополнительными байтами, добавленными БЧХ
    """
    def __init__(self, n, k):
        self.data = [0] * k
        self.index = 0
        self.coder = galois.BCH(n, k)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= 2 ** len(self.data):
            self.__iter__()
            raise StopIteration
        else:
            binary_num = format(self.index, '0' + str(len(self.data)) + 'b')
            for i in range(len(binary_num)):
                self.data[i] = int(binary_num[i])
            self.index += 1
            return self.get_data()

    def get_data(self):
        x = np.array(self.coder.encode([self.data])).astype(np.int64)
        y = np.array(self.coder.encode([self.data])).astype(np.int64)
        return x, y
