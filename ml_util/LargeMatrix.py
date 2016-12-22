import os
import numpy as np
import ctypes
import csv
import linecache
import file_utility as futil
from functional_utility import check_none

from enum import Enum

class LargeMatrixMode(Enum):
    shadow = 0
    data_reader = 1
    

class LargeMatrix(object):
    """ This class maps a 2-dimensional array to a csv file without loading the full data.
        Useful for manipulating large data
    """

    def __init__(self, file_path_or_parent, indices=None, encoding='utf8', delimiter=",", next_line="\n"):
        super().__init__()
        check_none(file_path_or_parent=file_path_or_parent)

        if isinstance(file_path_or_parent, LargeMatrix):
            self.from_parent(file_path_or_parent, indices)
        else:
            if file_path_or_parent == "" or not isinstance(file_path_or_parent, str):
                raise ValueError("Argument \"file_path\" must be a valid filesystem path or a LargeMatrix instance to be used as a parent.")

            file_path = file_path_or_parent
            if not os.path.exists(file_path):
                file_path = file_path + '.csv'
                if not os.path.exists(file_path):
                    raise FileNotFoundError("The requested csv file not found on the given path or was not a valid csv file.")
                       
                
            self._file_path = file_path
            self._indices = indices
            self._encoding = encoding
            self._delimiter = delimiter
            self._next_line = next_line
            self.shape = self._get_shape()

    _file_path = ""
    _indices = None
    _encoding = "utf8"
    _delimiter = ","
    _next_line = "\n"
    _gen_counter = 0

    _mode = LargeMatrixMode.shadow

    shape = None
    file_shape = None

    def __len__(self):
        return self._get_shape()[0]

    def __getitem__(self, the_slices):
        check_none(the_slices=the_slices)

        x_slice = the_slices[0] if isinstance(the_slices, tuple) else slice(the_slices, the_slices + 1, None)
        y_slice = the_slices[1] if isinstance(the_slices, tuple) and len(the_slices) > 1 else slice(0,1,None)

        x_start = 0 if x_slice.start is None else x_slice.start
        x_stop = self.shape[0] if x_slice.stop is None else x_slice.stop
        x_slice = slice(x_start, x_stop, None)

        y_start = 0 if y_slice.start is None else y_slice.start
        y_stop = self.shape[1] if y_slice.stop is None else y_slice.stop
        y_slice = slice(y_start, y_stop, None)

        dimensions = ((x_slice.stop - x_slice.start), (y_slice.stop - y_slice.start))
        
        if (dimensions[0] > self.shape[0] or dimensions[1] > self.shape[1]):
            raise IndexError("Given slices was outside the boundry of the matrix.")       

        x = slice(self._indices[0].start + x_slice.start, self._indices[0].start + x_slice.stop, None)
        y = slice(self._indices[1].start + y_slice.start, self._indices[1].start + y_slice.stop, None)

        if self._mode == LargeMatrixMode.data_reader:
            result = []

            for row_no in range(x.start, x.stop - 1):
                row = self.get_row(row_no + 1, y)
                if (dimensions == (1,1)):
                    return row[0]
                else:
                    result.append(row)

            return result
        elif self._mode == LargeMatrixMode.shadow:
            result = LargeMatrix(self, indices=(x,y))
            return result
        else:
            raise NotImplementedError("The given mode was not implemeneted: {0}.".format(str(self._mode)))

    def __setitem__(self, the_slices, value):
        raise NotImplementedError("Set item is not supported for files since it is too expensive.")
        check_none(the_slices=the_slices)
        x_slice = the_slices[0]
        y_slice = the_slices[1] if not the_slices is None and len(the_slices) > 1 else slice(0,1,None)
        dimensions = ((x_slice.stop - x_slice.start), (y_slice.stop - y_slice.start))

        if isinstance(value, list) and len(value) != (x_slice.stop - x_slice.start):
            raise ValueError("The dimensions of indices and given list size do not match. Expected " + str(dimensions[0]) + "x" + str(dimensions[1]))
        else:
            raise ValueError("For 1-dimensional assignment, the given index must be non dimensional")

        with open(self._file_path, 'a') as csv_file:
            writer = csv.writer(csv_file)
            for row_no in range(x_slice.start, x_slice.stop):
                row = self._get_line(self._file_path, row_no + 1)
                elements = futil.array_from_csv_row(row, delimiter=self._delimiter)
                elements[row_no, y_slice] = value if dimensions == (1,1) else value[row_no, y_slice]
                writer.writerow(elements)

    def __eq__(self, other):
        return self._file_path == other._file_path and self._indices == other._indices

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        raise NotImplementedError("LargeMatrix does not support iteration. Use yeild instead.")

    def __next__(self):
        if self._gen_counter > (self._indices[0].stop - self._indices[0].start):
            raise StopIteration()

        self._gen_counter = self._gen_counter + 1
        return 




    def set_mode(self, mode):
        if (mode is None or not isinstance(mode, LargeMatrixMode)):
            raise ValueError("Argument \"mode\" must be a valid LargeMatrixMode object.")
        self._mode = mode

    def _get_shape(self):
        if self.shape is None:
            self.shape = (None, None)

            if self.file_shape is None or self.file_shape == (None, None):
                file_row_no = 0
                file_col_no = 0
                with open(path, "rU", encoding=self._encoding) as f:
                    file_row_no = sum(1 for x in f)

                if file_row_no <= 0:
                    file_col_no = 0
                else:
                    row = linecache.getline(self._file_path, 0)
                    file_col_no = len(futil.array_from_csv_row(row, delimiter = self._delimiter))

                self.file_shape = (file_row_no, file_col_no)

            if self._indices is None:
                self.shape = self.file_shape
            else:
                self.shape = (self._indices[0].stop - self._indices[0].start, self._indices[1].stop - self._indices[1].start)

        if self._indices is None:
            self._indices = (slice(0,self.shape[0],None), slice(0,self.shape[1],None))

        return self.shape

    def get_row(self, row_no, columns_slice):
        row = self._get_line(row_no)
        elements = futil.array_from_csv_row(row, delimiter=self._delimiter)
        return elements[columns_slice.start:columns_slice.stop]

    def reset_generator(self):
        self._gen_counter = 0

    def _get_line(self, line_no):
        line = linecache.getline(self._file_path, line_no)

        if (len(linecache.cache) > 1000):
            linecache.clearcache()

        return line
    
    def from_parent(self, parent, indices):
        check_none(parent=parent, indices=indices)
        if not isinstance(parent, LargeMatrix):
            raise ValueError("Parent argument must be a LargeMatrix object.")

        if not isinstance(indices, tuple) or not len(indices) == 2 \
                or not isinstance(indices[0], slice) or not isinstance(indices[1], slice):
            raise ValueError("indices argument must be a tuple of slices.")

        self._file_path = parent._file_path
        self._delimiter = parent._delimiter
        self._encoding = parent._encoding
        self._next_line = parent._next_line
        self.file_shape = parent.file_shape
        self._indices = indices
        self.shape = self._get_shape()


    def transpose(self):
        pass

    def dot(self, target):
        pass

    def substract(self, target):
        pass

    def power(self, target):
        pass

    def multiply(self, target):
        pass

    def add(self, target):
        pass

    def break_to_batches(self, batch_size):
        pass


path = "D:/Projects/VS2015/PFTD.NLP/PFTD.NLP/information_verification/tokenization/output_data/tokenization_ngram_features_old.csv"
arr = LargeMatrix(path)
print(arr[0])
arr.set_mode(LargeMatrixMode.data_reader)
print(arr[0])
arr.set_mode(LargeMatrixMode.shadow)
print(arr[0:3,1:4])
arr2 = arr[0:3,1:4]
arr2.set_mode(LargeMatrixMode.data_reader)
arr.set_mode(LargeMatrixMode.data_reader)
print(arr[0:3,1:4])
print("----------------------")
print(arr2[0:1,2:3])
#arr[1:5,1:4] = []