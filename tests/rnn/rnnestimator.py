import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.contrib.estimator import RNNClassifier
import tensorflow.contrib.feature_column as contrib_feature_column
import test_utils

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
#print( "Vocabulary: " , vocabulary )

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


def input_fn(n_repetitions = 1) -> tf.data.Dataset:
    """
    Returns the text as char array

    Args:
        n_repetitions: Number of times to repeat the text
    """

    # The dataset
    ds = tf.data.Dataset.from_tensors( (inputs,outputs) )
    #ds = tf.data.Dataset.from_tensor_slices( (inputs,outputs) )

    # If we are training, we will need a lot of samples, not only four. So, repeat the 4 values
    if n_repetitions > 1:
        ds = ds.repeat(n_repetitions)

    # ds = ds.shuffle( 1000 )
    # ds = ds.batch(16)
    
    return ds

#test_utils.debug_ds( input_fn() , True )

#character_column = feature_column.categorical_column_with_vocabulary_list( 'character' , vocabulary_list = vocabulary )
character_column = contrib_feature_column.sequence_categorical_column_with_vocabulary_list( 'character' , vocabulary_list = vocabulary )
indicator_column = feature_column.indicator_column( character_column )
feature_columns = [ indicator_column ]

estimator = RNNClassifier(
    sequence_feature_columns=feature_columns,
    num_units=[7], cell_type='lstm', model_dir='./model', 
    n_classes=10, label_vocabulary=vocabulary)

repeated_input_fn = lambda:input_fn(100)
while test_utils.accuracy(estimator, repeated_input_fn) < 1.0:
    estimator.train(input_fn = repeated_input_fn)
    test_utils.accuracy(estimator, repeated_input_fn)


def predict( sequence ):

    def input_fn_predict():
        x = ({ 'character' : [ sequence ] })
        return tf.data.Dataset.from_tensors( x )
        
    result = estimator.predict( input_fn=input_fn_predict )
    #result = estimator.predict( input_fn=input_fn )
    #print(result)
    print("-----")
    print("Input: " , sequence )
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


# Inputs and outputs
# def predict_fn():
#     inputs = { 'character': [] }
#     outputs =  []
#     for i in range(2, 3):
#         sequence = text[i : i + SEQUENCE_LENGHT]
#         sequence_output = text[i + SEQUENCE_LENGHT : i + SEQUENCE_LENGHT+1]
#         #print(sequence , sequence_output)
#         inputs['character'].append( list(sequence) )
#         outputs.append(sequence_output)

#     print(inputs)
#     print(outputs)

#     ds = tf.data.Dataset.from_tensors( (inputs) )
#     return ds

# result = estimator.predict( input_fn=predict_fn )
# print("-----")
# for r in result:
#     print(r)
#     print('Output:', r['class_ids'])
# print("-----")
