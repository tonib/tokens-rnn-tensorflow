import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.contrib.estimator import RNNClassifier
import tensorflow.contrib.feature_column as contrib_feature_column
import test_utils
from typing import List

# Test to make a char RNN predictor

# Simple sample text to learn: repetitions of "0123456789"
text=""
for _ in range(10):
    text += "0123456789"

# Sequence length that will be feeded to the network
SEQUENCE_LENGHT = 7

# The real vocabulary:
vocabulary = list( set(text) )

# As far I know, Tensorflow RNN estimators don't support variable length sequences, so I'll use the char "_" as padding
# Maybe it's supported, right now I dont' know how
vocabulary.append('_')

# Important! Otherwise, with different executions, the list can be in different orders (really)
vocabulary.sort()

def pad_sequence( text_sequence : str ) -> List[str]:
    """
    Pads the text_sequence lenght to a minimum length of SEQUENCE_LENGHT, and returns it as a List of characters

    As far I know, Tensorflow RNN estimators don't support variable length sequences, so I'll use the char "_" as padding.
    If text_sequence has a len(text_sequence) < SEQUENCE_LENGHT, the text will be padded as "_...text_sequence", up to SEQUENCE_LENGHT characters.
    
    Args:
        text_sequence: The text to pad 

    Retunrs:
        The "text_sequence", padded with "_" characters, as a characters List
    """

    l = len(text_sequence)
    if l < SEQUENCE_LENGHT:
        # Pad string: "__...__text_sequence"
        text_sequence = text_sequence.rjust( SEQUENCE_LENGHT , '_')
    
    # Return the text as a characters list
    return list(text_sequence)

# Train input and outputs
inputs = { 'character': [] }
outputs =  []

def prepare_train_sequences_length(seq_length : int):
    """
    Prepare sequences of a given length

    Args:
        lenght: Length of sequences to prepare
    """
    for i in range(0, len(text) - seq_length):
        sequence = text[i : i + seq_length]
        sequence_output = text[i + seq_length : i + seq_length+1]
        inputs['character'].append( pad_sequence(sequence) )
        outputs.append(sequence_output)


# Prepare sequences of a range of lengths from 1 to 7 characters
for sequence_length in range(1, 8):
    prepare_train_sequences_length(sequence_length)

print("N. train sequences: ", len(inputs['character']))

def input_fn(n_repetitions = 1) -> tf.data.Dataset:
    """
    Returns the text as char array

    Args:
        n_repetitions: Number of times to repeat the inputs
    """

    # The dataset
    ds = tf.data.Dataset.from_tensor_slices( (inputs,outputs) )

    # Repeat inputs n times
    if n_repetitions > 1:
        ds = ds.repeat(n_repetitions)

    ds = ds.shuffle( 1000 )
    ds = ds.batch(4)
    
    return ds

# The single character sequence 
character_column = contrib_feature_column.sequence_categorical_column_with_vocabulary_list( 'character' , vocabulary_list = vocabulary )
indicator_column = feature_column.indicator_column( character_column )
feature_columns = [ indicator_column ]

# The estimator
estimator = RNNClassifier(
    sequence_feature_columns=feature_columns,
    num_units=[7], cell_type='lstm', model_dir='./model', 
    n_classes=10, label_vocabulary=vocabulary)

# Traing until all train set is learned
while test_utils.accuracy(estimator, input_fn) < 1.0:
    print("Training...")
    estimator.train(input_fn=input_fn)

def predict( text : str ):
    """
    Predicts and print the next character after a given sequence

    Args:
        text: The input sequence text
    """

    result = estimator.predict( input_fn=lambda:tf.data.Dataset.from_tensors( ({ 'character' : [ pad_sequence(text) ] }) ) )
    print("-----")
    print("Input sequence: " , text )
    for r in result:
        #print("Prediction: " , r)
        print('Class output: ', r['class_ids'])
    print("-----")

# Some predictions in the train set (memory)
predict( '0123456' )
predict( '1234567' )
predict( '2345678' )
predict( '3456789' )
predict( '4567890' )
predict( '3' )
predict( '5678' )

# Some predictions out the train set (generalization)
predict( '2345678901' )
predict( '6789012345678' )
predict( '9012345678901234567890123456789012' )
predict( '0123456789012345678901234567890123456789' )
