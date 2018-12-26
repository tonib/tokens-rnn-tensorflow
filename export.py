import argparse
from input_data import InputData
from model import Model
import tensorflow as tf
import os
from cmdline import parse_command_line

# Get commnad line arguments
args = parse_command_line()

# Get data and model
input_data = InputData(args)
model = Model(input_data, args)

def serving_input_receiver_fn():
    # TODO: Check if placeholder with variable input lenght  is allowed, for variable input sequences
    # It seems the shape MUST include the batch size (the 1)
    x = tf.placeholder(dtype=tf.string, shape=[1, Model.SEQUENCE_LENGHT], name='character')
    #print("Input shape: " , x)
    inputs =  {'character': x }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

path = os.path.join( args.data_dir , 'exportedmodel' )
print("Exporting to" , path)
model.estimator.export_savedmodel( path , serving_input_receiver_fn, strip_default_attrs=True)
