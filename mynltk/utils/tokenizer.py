import re
class Tokenizer:
    """ A tokenizer that splits text into tokens using a regular expression.
    Curently the method used is the simplest possible one.
    """

    def __init__(self, pattern=None, lowercase=False):
        self.pattern = pattern or r"\w+|[^\w\s]"
        self.lowercase = lowercase
    token_specification = [
        ('NUMBER',   r'\d+(\.\d*)?'),  # Integer or decimal number
        ('ASSIGN',   r':='),           # Assignment operator
        ('END',      r';'),            # Statement terminator
        ('ID',       r'[A-Za-z]+'),    # Identifiers
        ('OP',       r'[+\-*/]'),      # Arithmetic operators
        ('NEWLINE',  r'\n'),           # Line endings
        ('SKIP',     r'[ \t]+'),       # Skip over spaces and tabs
        ('MISMATCH', r'.'),            # Any other character
    ]

    def tokenize(self, text: str, char_based=False, lowercase=True) -> list[str]:
        """For now we use a simple tokenizer"""
        if self.lowercase:
            text =  [t.lower() for t in text] # better than text.lower()
        if char_based:
            return list(text)
        return self.simple_tokenize(text)
    
    def simple_tokenize(self, text: str) -> list[str]:
        """The simplest possible implemenation of a tokenizer"""
        return re.findall(self.pattern, text)