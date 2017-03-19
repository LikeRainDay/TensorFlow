import tensorflow as  tf
import input_data

minst = input_data.read_data_sets("MINST_data", one_hot=True)
