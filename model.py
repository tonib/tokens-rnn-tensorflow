from input_data import InputData
from tensorflow.contrib.estimator import RNNClassifier
import tensorflow as tf
import tensorflow.contrib.feature_column as contrib_feature_column
import tensorflow.feature_column as feature_column
import os

class Model:

    # Sequence length that will be feeded to the network
    SEQUENCE_LENGHT = 100

    def __init__(self, input_data : InputData, args: object, export_model_path : str = None):
        
        self.input_data = input_data

        if not export_model_path:
            self._create_estimator(args)
        else:
            # Import model
            self.predict_fn = tf.contrib.predictor.from_saved_model(export_model_path , signature_def_key='predict')


    def _create_estimator(self, args: object):

        # The single character sequence 
        character_column = contrib_feature_column.sequence_categorical_column_with_vocabulary_list( 'character' , 
            vocabulary_list = self.input_data.vocabulary )
        indicator_column = feature_column.indicator_column( character_column )
        feature_columns = [ indicator_column ]

        path = os.path.join( args.data_dir , 'model' )

        self.estimator = RNNClassifier(
                sequence_feature_columns=feature_columns,
                num_units=[64, 64], cell_type='gru', 
                optimizer=tf.train.AdamOptimizer,
                model_dir=path, 
                n_classes=len( self.input_data.vocabulary ), 
                label_vocabulary=self.input_data.vocabulary)