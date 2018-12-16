import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.contrib.estimator import RNNClassifier
import tensorflow.contrib.feature_column as contrib_feature_column
import argparse
import test_utils

#tf.enable_eager_execution()

# Get commnad line arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--op', type=str, default='train',
                    help='"train" to train, "generate" to generate text, "debugds" to debug dataset')
args = parser.parse_args()
print( args.op )

# Test to prectict characters from the Quixote first chapters
with open( 'quixote.txt' , 'r')  as file:
    text = file.read()
#print(text)

MAX_TEXT_LENGHT = 2000000

# Right now all if I try to run this will all text, it eats all my computer memory and it hags. Avoid it:
if len(text) > MAX_TEXT_LENGHT:
    text = text[:MAX_TEXT_LENGHT]

# Sequence length that will be feeded to the network
SEQUENCE_LENGHT = 200

# The real vocabulary:
vocabulary = list( set(text) )

# Important! Otherwise, with different executions, the list can be in different orders (really)
vocabulary.sort()

print("Vocabulary: " , vocabulary)

# Train input and outputs
# inputs = { 'character': [] }
# outputs =  []
# for i in range(0, len(text) - SEQUENCE_LENGHT):
#     sequence = text[i : i + SEQUENCE_LENGHT]
#     sequence_output = text[i + SEQUENCE_LENGHT : i + SEQUENCE_LENGHT+1]
#     inputs['character'].append( list(sequence) )
#     outputs.append(sequence_output)
#     #print("Sequence: " , sequence , ", Output: " , sequence_output)

# print("N. train sequences: ", len(inputs['character']))

def generator():
    print( 'Generator started' )
    for i in range(0, len(text) - SEQUENCE_LENGHT):
        sequence = text[i : i + SEQUENCE_LENGHT]
        sequence_output = text[i + SEQUENCE_LENGHT : i + SEQUENCE_LENGHT+1]
        # inputs['character'].append( list(sequence) )
        # outputs.append(sequence_output)
        #print( 'Sequence: ', sequence, ', output: ' , sequence_output )
        yield ( { 'character' : list(sequence) } , sequence_output )
    print( 'Generator finished' )

def input_fn(n_repetitions = 1) -> tf.data.Dataset:
    """
    Returns the text as char array

    Args:
        n_repetitions: Number of times to repeat the inputs
    """

    # The dataset
    #ds = tf.data.Dataset.from_tensor_slices( (inputs,outputs) )
    ds = tf.data.Dataset.from_generator( generator=generator, 
        output_types=( { 'character' : tf.string } , tf.string ),
        output_shapes=( { 'character' : (SEQUENCE_LENGHT,) } , () )
    )

    # Repeat inputs n times
    if n_repetitions > 1:
        ds = ds.repeat(n_repetitions)

    ds = ds.shuffle( 1000 )
    ds = ds.batch(64)
    
    return ds

# The single character sequence 
character_column = contrib_feature_column.sequence_categorical_column_with_vocabulary_list( 'character' , vocabulary_list = vocabulary )
indicator_column = feature_column.indicator_column( character_column )
feature_columns = [ indicator_column ]

# The estimator
estimator = RNNClassifier(
    sequence_feature_columns=feature_columns,
    num_units=[200], cell_type='gru', 
    optimizer=tf.train.AdamOptimizer,
    model_dir='./model', 
    n_classes=len(vocabulary), 
    label_vocabulary=vocabulary)

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

        # Instead returning the most probable character, get a random sample (see https://www.tensorflow.org/tutorials/sequences/text_generation):
        g_1 = tf.Graph()
        with g_1.as_default():
            with tf.Session():
                logits = tf.reshape( r['logits'] , ( 1 , -1 ) )
                idx = tf.multinomial( logits=logits , num_samples=1)
                idx = tf.squeeze(idx,axis=-1).eval()
                print("Multinomial idx: " , idx, ", Character: " , vocabulary[idx[0]] )
                # Comment this line to return the most probable char
                #c = vocabulary[idx[0]]

        return c
    print("-----")
    return '?'

def predict_text( text : str ) -> str:
    result = text
    for _ in range(50):
        result += predict_char( result )
    print(result)

if args.op == 'train':
    # Training loop
    while True:
        print("training...")
        estimator.train(input_fn=input_fn)
        print("evaluating...")
        test_utils.accuracy(estimator, input_fn, steps=100)
elif args.op == 'debugds':
    test_utils.debug_ds(input_fn() , True)
else:
    start_text = text[:SEQUENCE_LENGHT]
    print("start_text len:", len(start_text))
    predict_text( start_text )
    #predict_text( 'abc' )

