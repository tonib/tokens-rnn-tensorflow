import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.contrib.estimator import RNNClassifier
import tensorflow.contrib.feature_column as contrib_feature_column
import test_utils
from typing import List

#tf.enable_eager_execution()

# Test to make a char RNN predictor

# Simple text: repetitions of "0123456789"
text=""
for _ in range(10):
    text += "0123456789"

# Vocabulary:
vocabulary = list( set(text) )
# Important! Otherwise, with different executions, the list can be in different orders (really)
vocabulary.sort()

# Train input and outputs
inputs = { 'character': [] }
outputs =  []

def prepare_sequences_length(seq_length : int):
    """
    Prepare sequences of a given length

    Args:
        lenght: Length of sequences to prepare
    """
    for i in range(0, len(text) - seq_length):
        sequence = text[i : i + seq_length]
        sequence_output = text[i + seq_length : i + seq_length+1]
        inputs['character'].append( list(sequence) )
        outputs.append(sequence_output)


# Prepare sequences of a range of lengths
for sequence_length in range(7, 8):
    prepare_sequences_length(sequence_length)

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

character_column = contrib_feature_column.sequence_categorical_column_with_vocabulary_list( 'character' , vocabulary_list = vocabulary )
indicator_column = feature_column.indicator_column( character_column )
feature_columns = [ indicator_column ]

estimator = RNNClassifier(
    sequence_feature_columns=feature_columns,
    num_units=[7], cell_type='lstm', model_dir='./model', 
    n_classes=10, label_vocabulary=vocabulary)

while test_utils.accuracy(estimator, input_fn) < 1.0:
    print("Training...")
    estimator.train(input_fn = lambda:input_fn(100))

def predict( sequence : List[str] ):
    """
    Predicts and print the next character after a given sequence

    Args:
        sequence: The input sequence (list of characters)
    """

    result = estimator.predict( input_fn=lambda:tf.data.Dataset.from_tensors( ({ 'character' : [ sequence ] }) ) )
    print("-----")
    print("Input sequence: " , sequence )
    for r in result:
        print(r)
        print('Output:', r['class_ids'])
    print("-----")


predict( [ '0' , '1' , '2' , '3' , '4' , '5' , '6'] ) # OK
predict( [ '1' , '2' , '3' , '4' , '5' , '6', '7' ] ) # OK
predict( [ '2' , '3' , '4' , '5' , '6', '7' , '8' ] ) # OK
predict( [ '3' , '4' , '5' , '6', '7' , '8' , '9' ] ) # OK
predict( [ '4' , '5' , '6', '7' , '8' , '9' , '0' ] ) # OK
predict(  [ '3' ] ) # This fails
predict(  [ '5' , '6' , '7' , '8' ] ) # This fails
predict( [ '2' , '3' , '4' , '5' , '6', '7' , '8' , '9' , '0' , '1' ] ) # This fails
