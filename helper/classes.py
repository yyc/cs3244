import pickle

def extract_classes(path):
    pickle_off = open(path, "rb")
    pickle.load(pickle_off)
    pick2 = pickle.load(pickle_off)
    return pick2

if __name__ == '__main__':
    pik = extract_classes("../data/classes1M.pkl")
    print(pik)