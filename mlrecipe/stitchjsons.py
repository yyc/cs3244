import json

files = ["full_training_packet_part1.json",
         "full_training_packet_part2.json",
         "full_training_packet_part3.json",
         "full_training_packet_part4.json",
         "full_training_packet_part5.json",
         "full_training_packet_part6.json",
         "full_training_packet_part7.json",
         "full_training_packet_part8.json",
         "full_training_packet_part9.json",
         "full_training_packet_part10.json",
         "full_training_packet_part11.json"]

CONCATENATED_TRAINING_PACKET_NAME = "full_training.json"

concatenated_training_packet = []

for i in range(len(files)):
    with open(files[i]) as training_packet_i:
        print "Parsing {}".format(files[i])
        d = json.load(training_packet_i)
        concatenated_training_packet.extend(d)

with open(CONCATENATED_TRAINING_PACKET_NAME, 'w') as outfile:
    json.dump(concatenated_training_packet, outfile)
