from symspell.symspellpy import SymSpell  # import the module
import pathlib

class Spellcheck:
    
    sym_spell = SymSpell(2, 7)
      
    def correct_sentence(self, sentence):
        return self.sym_spell.word_segmentation(sentence).corrected_string  

class EnglishSpellCheck(Spellcheck):
    def __init__(self):
        if not self.sym_spell.create_dictionary("symspell/frequency_dictionary_en_82_765.txt"):
            print("English corpus file not found")
            return

class DutchSpellCheck(Spellcheck):
    def __init__(self):
        if not self.sym_spell.create_dictionary("symspell/nl_50k.txt"):
            print("Dutch corpus file not found")
            return

class FrenchSpellCheck(Spellcheck):
    def __init__(self):
        if not self.sym_spell.create_dictionary("symspell/fr_50k.txt"):
            print("French corpus file not found")
            return