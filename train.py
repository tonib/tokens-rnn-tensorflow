
from input_data import InputData
from model import Model
import argparse
import random
import tensorflow as tf
import tests.test_utils as test_utils

# Get commnad line arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='data/quixote', help='Model directory')
args = parser.parse_args()

# Get data and model
input_data = InputData(args)
model = Model(input_data, args)

TRAIN_SIZE = 8000

def train_generator():
    for _ in range(TRAIN_SIZE):
        sequence_start_idx = random.randint( 0 , len(input_data.text) - Model.SEQUENCE_LENGHT )
        input = input_data.get_sequence(sequence_start_idx, Model.SEQUENCE_LENGHT)
        output = input_data.get_sequence_output(sequence_start_idx, Model.SEQUENCE_LENGHT)
        yield ( { 'character' : input } , output )

def input_fn(evaluate=False) -> tf.data.Dataset:
    """
    Returns the text as char array

    Args:
        n_repetitions: Number of times to repeat the inputs
    """

    # The dataset
    ds = tf.data.Dataset.from_generator( generator=train_generator, 
        output_types=( { 'character' : tf.string } , tf.string ),
        output_shapes=( { 'character' : (Model.SEQUENCE_LENGHT,) } , () )
    )

    ds = ds.batch(64)
    ds = ds.prefetch(1)

    return ds

# Training loop
while True:
    print("training...")
    model.estimator.train(input_fn=input_fn)
    print("evaluating...")
    test_utils.accuracy(model.estimator, lambda:input_fn(True), steps=100)
