# WRT: thamsy's https://github.com/yyc/cs3244/blob/master/helper/classes.py
import pickle

def extract_classes(path):
    pickle_off = open(path, "rb")
    pickle.load(pickle_off)
    pick2 = pickle.load(pickle_off)
    return pick2

x = extract_classes("classes1M.pkl")
print x
