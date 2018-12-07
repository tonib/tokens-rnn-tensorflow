import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.estimator import DNNClassifier

# Example to learn a XOR function
# Use "python classifierestimator.py": Train and test the model. The model will be reused after the train
# Use rm -rf model to delete the current model, a new one will be created automatically

# TODO: Remove tf.constant() calls?
# TODO: estimator.predict is EXTREMELLY SLOW. It seems it rebuilds the graph each time is called (Use predictors?)


def input_fn(n_repetitions = 1) -> tf.data.Dataset:
    """
    Returns the XOR function dataset

    Args:
        n_repetitions: Number of times to repeat the XOR function
    """

    # The inputs. Each dict key is a dataset column. Each index is ONE column value. The array is the "batch" of inputs to train / estimate
    xor_x = {
        'x1': tf.constant( [ 0 , 1 , 0 , 1 ] ),
        'x2': tf.constant( [ 0 , 0 , 1 , 1 ] )
    }

    # The expected outputs ("labels"). Each index is ONE output. The array is the output "batch"
    xor_y =  tf.constant( [ 0 , 1 , 1 , 0 ] )

    # The dataset
    ds = tf.data.Dataset.from_tensors( (xor_x,xor_y) )

    # If we are training, we will need a lot of samples, not only four. So, repeat the 4 values
    if n_repetitions > 1:
        ds = ds.repeat(n_repetitions)
    
    return ds


# dataset input interesting columns (all!)
x1_column = feature_column.numeric_column( 'x1' )
x2_column = feature_column.numeric_column( 'x2' )

# The model
estimator = DNNClassifier(
    hidden_units=[6, 6],
    feature_columns=[x1_column, x2_column],
    model_dir='./model'
)

def accuracy() -> float:
    """
    Returns the current ratio of succesfully predicted outputs of XOR function: 0.0 = 0%, 1.0 = 100%
    """

    # Estimate a dataset with no repetitions (all the possible XOR inputs)
    result = estimator.evaluate( input_fn=input_fn )
    print("Evaluation: ", result)
    return result['accuracy']


# Train until we get a 100% accuracy. Train 1000 times all the inputs on each call
while accuracy() < 1.0:
    estimator.train( input_fn=lambda:input_fn(1000) )

def single_eval_input_dataset( x1 : int, x2 : int ) -> tf.data.Dataset:
    """
    Returns a dataset with a single input for the XOR (without the output!)

    Args:
        x1, x2: XOR inputs
    """

    xor_x = {
        'x1': tf.constant( [ x1 ] ) ,
        'x2': tf.constant( [ x2 ] )
    }
    return tf.data.Dataset.from_tensors(xor_x)


def print_eval(x1 : int, x2 : int):
    """
    Print the output prediction for a XOR input

    Args:
        x1, x2: XOR inputs
    """

    result = estimator.predict( input_fn=lambda:single_eval_input_dataset(x1,x2) )
    #print(result)
    for r in result:
        print('Input:', x1, x2)
        print('Output:', r['class_ids'])


print("Results")
print_eval(0, 0)
print_eval(0, 1)
print_eval(1, 0)
print_eval(1, 1)
