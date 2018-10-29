import json
import pickle

files = ["training_jsons/full_training_packet_part1.json",
         "training_jsons/full_training_packet_part2.json",
         "training_jsons/full_training_packet_part3.json",
         "training_jsons/full_training_packet_part4.json",
         "training_jsons/full_training_packet_part5.json",
         "training_jsons/full_training_packet_part6.json",
         "training_jsons/full_training_packet_part7.json",
         "training_jsons/full_training_packet_part8.json",
         "training_jsons/full_training_packet_part9.json",
         "training_jsons/full_training_packet_part10.json",
         "training_jsons/full_training_packet_part11.json"]

CONCATENATED_TRAINING_PACKET_NAME = "training_jsons/full_training.pkl"

concatenated_training_packet = []

for i in range(len(files)):
    with open(files[i]) as training_packet_i:
        print("Parsing {}".format(files[i]))
        d = json.load(training_packet_i)
        concatenated_training_packet.extend(d)


with open(CONCATENATED_TRAINING_PACKET_NAME, 'wb') as outfile:
    pickle.dump(concatenated_training_packet, outfile)
