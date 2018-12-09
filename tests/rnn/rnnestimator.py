import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.contrib.estimator import RNNClassifier
import tensorflow.contrib.feature_column as contrib_feature_column

# Test to make a char RNN predictor

# Simple text: repetitions of "0123456789"
text=""
for _ in range(10):
    text += "0123456789"

# Vocabulary:
vocabulary = list( set(text) )
vocabulary.sort()
#print( vocabulary )

# Prepare batches

# Inputs and outputs
SEQUENCE_LENGHT = 7
inputs = { 'character': [] }
outputs =  []
for i in range(0, len(text) - SEQUENCE_LENGHT):
    sequence = text[i : i + SEQUENCE_LENGHT]
    sequence_output = text[i + SEQUENCE_LENGHT : i + SEQUENCE_LENGHT+1]
    #print(sequence , sequence_output)
    inputs['character'].append( list(sequence) )
    outputs.append(sequence_output)

#print(outputs)

def input_fn(n_repetitions = 1) -> tf.data.Dataset:
    """
    Returns the text as char array

    Args:
        n_repetitions: Number of times to repeat the text
    """

    # The dataset
    ds = tf.data.Dataset.from_tensors( (inputs,outputs) )

    # If we are training, we will need a lot of samples, not only four. So, repeat the 4 values
    if n_repetitions > 1:
        ds = ds.repeat(n_repetitions)
    
    return ds

#print( input_fn() )

#character_column = feature_column.categorical_column_with_vocabulary_list( 'character' , vocabulary_list = vocabulary )
character_column = contrib_feature_column.sequence_categorical_column_with_vocabulary_list( 'character' , vocabulary_list = vocabulary )
indicator_column = feature_column.indicator_column( character_column )
feature_columns = [ indicator_column ]

estimator = RNNClassifier(
    sequence_feature_columns=feature_columns,
    num_units=[7], cell_type='gru', model_dir='./model', 
    n_classes=10, label_vocabulary=vocabulary)

# while True:
#     estimator.train(input_fn = input_fn)
#     result = estimator.evaluate( input_fn=lambda:input_fn(1000) )
#     print("Evaluation: ", result)

def predict( sequence ):
    result = estimator.predict( input_fn=lambda:tf.data.Dataset.from_tensors( { 'character' : sequence } ) )
    #print(result)
    for r in result:
        print(r)
        print('Output:', r['class_ids'])

#predict( [ '0' , '1' , '2' , '3' , '4' , '5' , '6'] )
#predict( [ '1' , '2' , '3' , '4' , '5' , '6', '7' ] )
#predict( [ '2' , '3' , '4' , '5' , '6', '7' , '8' ] )
#predict(  [ '5' ] )
#predict(  [ '5' , '6' , '7' , '8' ] )
#predict( [ '0' , '1' , '2' ] )
#predict( [ '1' ] )
predict( [ '2' , '3' , '4' , '5' , '6', '7' , '8' , '9' , '0' , '1' ] )
