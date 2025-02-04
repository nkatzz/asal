# A small test for the parsing of the sapienza LTLf2dfa
# conversion.
from src.asal_nesy.deepfa.utils import parse_sapienza_to_fa

fa = parse_sapienza_to_fa("G((blocked | tired) -> WX(~move))")
fa.dot().view()
