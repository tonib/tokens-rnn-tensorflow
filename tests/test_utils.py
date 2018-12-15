import tensorflow as tf

tf.enable_eager_execution()

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


def accuracy(estimator, input_fn) -> float:
    """
    Returns the current ratio of succesfully predicted outputs of XOR function: 0.0 = 0%, 1.0 = 100%
    """

    # Estimate a dataset with no repetitions (all the possible XOR inputs)
    result = estimator.evaluate( input_fn=input_fn )
    print("Evaluation: ", result)
    return result['accuracy']


