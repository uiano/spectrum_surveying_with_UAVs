import numpy as np
import tensorflow as tf
from collections import OrderedDict


def mat_argmax(m_A):
    """ returns tuple with indices of max entry of matrix m_A"""

    num_cols = m_A.shape[1]
    assert m_A.ndim == 2

    ind = np.argmax(m_A)

    row = ind // num_cols
    col = ind % num_cols
    return (row, col)


def mat_argmin(m_A):
    """ returns tuple with indices of min entry of matrix m_A"""

    num_cols = m_A.shape[1]
    assert m_A.ndim == 2

    ind = np.argmin(m_A)

    row = ind // num_cols
    col = ind % num_cols
    return (row, col)


def print_time(start_time, end_time):
    td = end_time - start_time
    hours = td.seconds // 3600
    reminder = td.seconds % 3600
    minutes = reminder // 60
    seconds = (td.seconds - hours * 3600 -
               minutes * 60) + td.microseconds / 1e6
    time_str = ""
    if td.days:
        time_str = "%d days, " % td.days
    if hours:
        time_str = time_str + "%d hours, " % hours
    if minutes:
        time_str = time_str + "%d minutes, " % minutes
    if time_str:
        time_str = time_str + "and "

    time_str = time_str + "%.3f seconds" % seconds
    #set_trace()
    print("Elapsed time = ", time_str)


def empty_array(shape):
    return np.full(shape, fill_value=None, dtype=float)


def project_to_interval(x, a, b):
    assert a <= b
    return np.max([np.min([x, b]), a])


def watt_to_dbm(array):
    assert (array > 0).all()

    return 10 * np.log10(array) + 30


def dbm_to_watt(array):
    return 10**((array - 30) / 10)


def natural_to_dB(array):  # array is power gain
    return 10 * np.log10(array)


def dB_to_natural(array):  # array is power gain
    return 10**(array / 10)


def save_l_var_vals(l_vars):
    # returns a list of tensors with the values of the
    # variables in the list l_vars.
    l_vals = []
    for var in l_vars:
        l_vals.append(tf.convert_to_tensor(var))
    return l_vals


def restore_l_var_vals(l_vars, l_vals):

    assert len(l_vars) == len(l_vals)
    # assigns the value l_vals[i] to l_vars[i]
    for var, val in zip(l_vars, l_vals):
        var.assign(val)


class FifoUniqueQueue():
    """FIFO Queue that does not push a new element if it is already in the queue. Pushing an element already in the queue does not change the order of the queue.
    It seems possible to implement this alternatively as a simple list.
"""
    def __init__(self):
        self._dict = OrderedDict()

    def put(self, key):
        self._dict[key] = 0  # dummy value, for future usage

    def get(self):
        # Returns oldest item
        return self._dict.popitem(last=False)[0]

    def empty(self):
        return len(self._dict) == 0


def list_are_close(list1, list2):
    """
    Args:
          `list1` can be a tensor or a list of tensors
          `list2` can be a tensor or a list of tensors

    Returns:
        True if list1 and list2 are the same.
    """
    assert type(list1) == type(list2)

    def compare(x, y):
        # return tf.abs(x - y)
        return tf.math.reduce_all(np.core.numeric.isclose(x, y, rtol=1e-6, atol=0,
                                       equal_nan=True))

    if type(list1) != list:

        return compare(list1, list2)

    else:

        return tf.reduce_all([compare(x, y) for x, y in zip(list1, list2)])


# a = np.random.random(size=(2,3,4,1))
# b = np.random.random(size=(2,3,4,1))
# l_1 = [a, b]
# l_2 = [a, a]
# # l_3 = [l_1,  l_1, l_2]
# #
# # l_4 =[l_2, l_1, l_1]
# # l_5 = [l_3, l_4]
# # l_6 = [l_3, l_3]
# e = list_are_close(l_1, l_2)
# #
# print( e)