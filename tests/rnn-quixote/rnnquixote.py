import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.contrib.estimator import RNNClassifier
import tensorflow.contrib.feature_column as contrib_feature_column
import argparse

# Get commnad line arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--op', type=str, default='train',
                    help='"train" to train, "generate" to generate text')
args = parser.parse_args()
print( args.op )

# Test to prectict characters from the Quixote first chapters
with open( 'quixote.txt' , 'r')  as file:
    text = file.read()
#print(text)

# Sequence length that will be feeded to the network
SEQUENCE_LENGHT = 200

# The real vocabulary:
vocabulary = list( set(text) )

# Important! Otherwise, with different executions, the list can be in different orders (really)
vocabulary.sort()

print("Vocabulary: " , vocabulary)

# Train input and outputs
inputs = { 'character': [] }
outputs =  []

for i in range(0, len(text) - SEQUENCE_LENGHT):
    sequence = text[i : i + SEQUENCE_LENGHT]
    sequence_output = text[i + SEQUENCE_LENGHT : i + SEQUENCE_LENGHT+1]
    inputs['character'].append( sequence )
    outputs.append(sequence_output)
    #print("Sequence: " , sequence , ", Output: " , sequence_output)

print("N. train sequences: ", len(inputs['character']))

# The single character sequence 
character_column = contrib_feature_column.sequence_categorical_column_with_vocabulary_list( 'character' , vocabulary_list = vocabulary )
indicator_column = feature_column.indicator_column( character_column )
feature_columns = [ indicator_column ]

# The estimator
estimator = RNNClassifier(
    sequence_feature_columns=feature_columns,
    num_units=[200], cell_type='lstm', model_dir='./model', 
    n_classes=len(vocabulary), label_vocabulary=vocabulary)

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
    ds = ds.batch(8)
    
    return ds

def accuracy(estimator, input_fn) -> float:
    """
    Returns the current ratio of succesfully predicted outputs of XOR function: 0.0 = 0%, 1.0 = 100%
    """

    # Estimate a dataset with no repetitions (all the possible XOR inputs)
    result = estimator.evaluate( input_fn=input_fn )
    print("Evaluation: ", result)
    return result['accuracy']

def predict_char( text : str ) -> str:
    """
    Predicts and print the next character after a given sequence

    Args:
        text: The input sequence text
    """

    result = estimator.predict( input_fn=lambda:tf.data.Dataset.from_tensors( ({ 'character' : [ list(text) ] }) ) )
    print("-----")
    print("Input sequence: '" , text , "'")
    for r in result:
        print('Prediction: ' , r)
        c = r['classes'][0].decode( 'utf-8' )
        print( 'Class output: "', c , '"')
        return c
    print("-----")

def predict_text( text : str ) -> str:
    result = text
    for _ in range(10):
        result += predict_char( result )
    print(result)

if args.op == 'train':
    # Training loop
    print( 'Training...' )
    while True:
        estimator.train(input_fn=input_fn)
        accuracy(estimator, input_fn)
else:
    predict_text( 'En un lugar de la Mancha' )
    #predict_text( 'abc' )

