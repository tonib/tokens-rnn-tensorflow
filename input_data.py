from enum import Enum
import os
from typing import List

class TokenMode(Enum):
    """ Work mode: Predict characters or words """
    CHARACTER = 1
    WORD = 2

class InputData:
    """ The train data: A text file """

    #def __init__(self, data_file_path : str , token_mode : TokenMode , sequence_length : int ):
    def __init__(self, args : object ):

        # Input file file
        path = os.path.join( args.data_dir , 'input.txt' )

        # The text to train / predict
        print("Reading", path)
        with open( path , 'r')  as file:
            self.text = file.read()

        # Text vocabulary
        self.vocabulary = list( set(self.text) )
        # Important! Otherwise, with different executions, the list can be in different orders (really)
        self.vocabulary.sort()

    def get_sequence( self, sequence_start_idx : int , sequence_length : int ) -> List[str]:
        return list( self.text[sequence_start_idx : sequence_start_idx + sequence_length] )

    def get_sequence_output( self, sequence_start_idx : int , sequence_length : int ) -> str:
        return self.text[sequence_start_idx + sequence_length : sequence_start_idx + sequence_length+1] 


