import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.contrib.estimator import RNNClassifier
import tensorflow.contrib.feature_column as contrib_feature_column
import argparse

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

# Right now all if I try to run this will all text, it eats all my computer memory and it hags. Avoid it:
if len(text) > 1000:
    text = text[:1000]

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
    inputs['character'].append( list(sequence) )
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
    num_units=[200], cell_type='gru', 
    optimizer=tf.train.AdamOptimizer,
    model_dir='./model', 
    n_classes=len(vocabulary), 
    label_vocabulary=vocabulary)

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
    ds = ds.batch(64)
    
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

        # Instead returning the most probable character, get a random sample (see https://www.tensorflow.org/tutorials/sequences/text_generation):
        g_1 = tf.Graph()
        with g_1.as_default():
            with tf.Session():
                logits = tf.reshape( r['logits'] , ( 1 , -1 ) )
                idx = tf.multinomial( logits=logits , num_samples=1)
                idx = tf.squeeze(idx,axis=-1).eval()
                print("Multinomial idx: " , idx, ", Character: " , vocabulary[idx[0]] )
                c = vocabulary[idx[0]]

        return c
    print("-----")
    return '?'

def predict_text( text : str ) -> str:
    result = text
    for _ in range(50):
        result += predict_char( result )
    print(result)

def debug_ds(ds, print_ds=True):
    if print_ds:
        print(ds)
        print()

    it = ds.make_one_shot_iterator()
    n_elements = 0
    while True:
        try:
            v = it.get_next()
        except:
            break
        
        n_elements += 1
        if print_ds:
            print(v)
            # print(v[0])
            # print(v[1])
            print()

    print("N.elements: ", n_elements)

if args.op == 'train':
    # Training loop
    print( 'Training...' )
    while True:
        estimator.train(input_fn=input_fn)
        accuracy(estimator, input_fn)
elif args.op == 'debugds':
    debug_ds(input_fn() , True)
else:
    start_text = text[:SEQUENCE_LENGHT]
    print("start_text len:", len(start_text))
    predict_text( start_text )
    #predict_text( 'abc' )

