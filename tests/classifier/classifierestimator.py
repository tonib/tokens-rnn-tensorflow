import tensorflow as tf
import tensorflow.feature_column as feature_column
from tensorflow.estimator import DNNClassifier
import time


def input_fn():

    xor_x = {
        'x1': tf.constant( [ 0 , 1 , 0 , 1 ] ),
        'x2': tf.constant( [ 0 , 0 , 1 , 1 ] )
    }

    xor_y =  tf.constant( [ 0 , 1 , 1 , 0 ] )

    ds = tf.data.Dataset.from_tensors( (xor_x,xor_y) )
    ds = ds.repeat(5000)
    return ds


x1_column = feature_column.numeric_column( 'x1' )
x2_column = feature_column.numeric_column( 'x2' )

estimator = DNNClassifier(
    hidden_units=[6, 6],
    feature_columns=[x1_column, x2_column],
    model_dir='./model'
)

def accuracy():
    result = estimator.evaluate( input_fn=input_fn )
    print(result)
    return result['accuracy']

while accuracy() < 1.0:
    estimator.train( input_fn=input_fn )

def eval_input( x1, x2 ):
    xor_x = {
        'x1': tf.constant( [ x1 ] ) ,
        'x2': tf.constant( [ x2 ] )
    }
    return tf.data.Dataset.from_tensors(xor_x)

def print_eval(x1, x2):
    result = estimator.predict( input_fn=lambda:eval_input(x1,x2) )
    #print(result)
    for r in result:
        print('Input:', x1, x2)
        print(r['class_ids'])

print("Results")
print_eval(0, 0)
print_eval(0, 1)
print_eval(1, 0)
print_eval(1, 1)
