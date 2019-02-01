from enum import Enum
import os
from typing import List

class InputData:
    """ The train data: A text file """

    #def __init__(self, data_file_path : str , token_mode : TokenMode , sequence_length : int ):
    def __init__(self, args : object ):

        # Input file file
        path = os.path.join( args.data_dir , 'input.txt' )

        # The text to train / predict
        print("Reading", path)
        with open( path , 'r', encoding='utf-8')  as file:
            self.text = file.read()

        self.word_mode = False
        if args.mode == 'word':
            # Words vocabulary. Store the text as a words list
            self.word_mode = True
            self.text = self.text.split()

        # Text vocabulary
        self.vocabulary = list( set(self.text) )

        # Important!
        self.vocabulary.sort()

        print( "Vocabulary length:", len(self.vocabulary) )
        print( "Text length:", len(self.text) , "tokens")

        #print( self.vocabulary )

    def get_sequence( self, sequence_start_idx : int , sequence_length : int ) -> List[str]:
        return list( self.text[sequence_start_idx : sequence_start_idx + sequence_length] )

    def get_sequence_output( self, sequence_start_idx : int , sequence_length : int ) -> str:
        output = self.text[sequence_start_idx + sequence_length : sequence_start_idx + sequence_length+1]
        if self.word_mode:
            output = output[0]
        return output

