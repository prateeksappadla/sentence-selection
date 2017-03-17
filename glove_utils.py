import json
import pickle

def createGloveDict(glove_embedding_file):
	vocab_dict = {}
	ind = 0
	with open(glove_embedding_file,'r') as glove_file:
		for line in glove_file:
			word = line.split()[0]
			vocab_dict[word] = ind
			ind += 1
	return vocab_dict		

glove_embedding_file = '/Users/prateek/Documents/qasystem/data_preprocessing/squad_embed_42791.txt'	
vocab_dict = createGloveDict(glove_embedding_file)

with open('glovedict_pickle.p','wb') as pickle_file:	
	pickle.dump(vocab_dict,pickle_file)

with open('glovedict_json.json', 'w') as json_file:
	json.dump(vocab_dict,json_file)	

# with open('glovedict_json.json') as dict_json_file:
# 	vocab_dict_loaded = json.load(dict_json_file)

with open('glovedict_pickle.p','rb') as dict_pickle_file:
	vocab_dict_loaded = pickle.load(dict_pickle_file)

print vocab_dict_loaded['bowl']	
