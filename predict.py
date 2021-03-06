import argparse
from input_data import InputData
from model import Model
import os
import tensorflow as tf
from tensorflow.contrib.predictor.predictor import Predictor
from cmdline import parse_command_line

def create_choose_random_char_tensor() -> tf.Tensor:
    """ Define Tensorflow ops to choose random character with temperature """

    logits_input = tf.placeholder( tf.double , [ len(input_data.vocabulary) ] )
    temperature_input = tf.placeholder( tf.double , [] )

    logits = tf.reshape( logits_input , ( 1 , -1 ) )
    logits = tf.divide( logits , temperature_input )
    char_idx = tf.multinomial( logits=logits , num_samples=1)
    char_idx = tf.squeeze(char_idx,axis=-1)
    return  { 'logits' : logits_input , 'temperature' : temperature_input , 'op' : char_idx }

def choose_random_char( predictions , temperature : float = -1 ) -> str:
    if temperature < 0:
        return predictions['classes'][0][0].decode( 'utf-8' )

    # Temperature: https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally
    # = 1: Original probabilities
    # -> ∞:  all [samples] have nearly the same probability
    # -> 0+: the probability of the [sample] with the highest expected reward tends to 1

    if not hasattr( choose_random_char , '_ops' ):
        # Define Tensorflow ops to choose random character ("static" variable, ughs)
        choose_random_char._ops = create_choose_random_char_tensor()

    with tf.Session() as session:
        idx = session.run( choose_random_char._ops['op'] , 
            { 
                choose_random_char._ops['logits'] : predictions['logits'][0] , 
                choose_random_char._ops['temperature'] : temperature 
            }
        )
        #print( idx )
        c = input_data.vocabulary[idx[0]]
        #print( c )
        return c

def predict_char_predictor( predict_fn : Predictor , input_sequence ) -> str:
    #print("Input: " , input_sequence)
    predictions = predict_fn( { 'character': [ input_sequence ] } )
    return choose_random_char( predictions , 0.3 )

# Get commnad line arguments
args = parse_command_line()

# Get data
input_data = InputData(args)

# Exports directory
exports_path = os.path.join( args.data_dir , 'exportedmodel' )

# Get latest export. Exports are versioned by a timestamp
latest_export = ''
max_timestamp = 0
for export_dir in os.listdir( exports_path ):
    try:
        timestamp = int(export_dir)
    except:
        timestamp = 0

    if timestamp > max_timestamp:
        max_timestamp = timestamp
        latest_export = export_dir
# The full path
latest_export = os.path.join( exports_path , latest_export )
print("Using export from" , latest_export)

# Get data from exported model
model = Model(input_data, args, latest_export )

start_sequence = input_data.get_sequence(0, Model.SEQUENCE_LENGHT)

# TODO: Check if placeholder with variable input lenght  is allowed, for variable input sequences
result = start_sequence
print( start_sequence )
next_sequence = start_sequence
while True:
    new_token = predict_char_predictor( model.predict_fn, next_sequence )
    #print( 'New token:"' + new_token + '"' )
    if input_data.word_mode:
        print( ' ' , end='')
    print( new_token, end='', flush=True)
    next_sequence = next_sequence[1:]
    next_sequence.append( new_token )
