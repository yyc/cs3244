import pickle

f = open('data/classes1M.pkl', 'rb')

_ = pickle.load(f)
classes = pickle.load(f)



# now classes is a dict with id -> food
#example:
for i in range(20):
  print("{}: {}".format(i, classes[i]))
