from keras.models import Sequential, model_from_json
from keras.layers import Dense
import numpy
import json
print(json.dumps(json.loads(open('model.json').read())))