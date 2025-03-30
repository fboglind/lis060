"""bpencoder.py"""
import tokenizer

class BPEncoder:
    """
    Preprocessing:
    The algorithm is usually run inside words (not merging across word boundaries),
    so the input corpus is first white-space-separated to give a set of strings, each corre
    sponding to the characters of a word, plus a special end-of-word symbol"""

    def whitespace_separate(self,C:str)->str:
        """Separates characters in a string with whitespace and adds end of word symbol: '_'"""
        C.replace(" ", "_")
        c_list=list(C)
        return str(c_list)






    def learn_encoding(self, C, k):
        """A method that takes a path to a text corpus (C) and an integer
        (k) indicating the desired vocabulary size, and uses the BPE algorithm
        to compute such a vocabulary"""

        V=set() # begins with a vocabulary that is just the set of all individual characters.
        # Examine the training corpus
        
        for i in range(k):
            #chooses the two symbols that are most frequently adjacent (say ‘A’, ‘B’)

            
        return V        
        

    def apply_encoding(self, text:str)->str:
        """A method that reads a text file and produces a tokenized text, using
        the BPE vocabulary produced by the previous method."""
        pass
    def load_encoding(self):
        """ Functionality to load the BPE vocabulary from a file, in order
        to avoid having to run the (slow) BPE algorithm more than once. """
        pass

    def save_encoding(self, filname:str):
        """ Functionality to save the BPE vocabulary to a file, in order
        to avoid having to run the (slow) BPE algorithm more than once. """
        pass