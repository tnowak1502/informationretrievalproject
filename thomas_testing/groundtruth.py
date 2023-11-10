import pickle
import json

# read binary data
with open('gt', 'rb') as binary_file:
    binary_data = binary_file.read()

# Process binary_data
# Unpickle (deserialize) the binary data
deserialized_data = pickle.loads(binary_data)



#example

print(deserialized_data["Mafia II"])
with open("groundtruth.json", "w") as fp:
    json.dump(deserialized_data, fp, indent=4)