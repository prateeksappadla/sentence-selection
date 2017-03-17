"""
Repeat for batch_size number of steps 
	select a context at random 
	select a question at random from the selected context
	*** retrieve the sentence and the index of the sentence in which the answer is present

	I get vectors of indices from pre-processing code
"""
import numpy as np 

batch_size = 32
  
# Get contexts from pre-processing code 
contexts = []

# Function to sample a random value
np.random.choice(contexts)

# Get maximum length sentence from all sentences
# max(len(x) for x in sentences)

def getContextBatch(data, batch_size):
	data_size = len(data) 
	shuffle_indices = np.random.permutation(np.arange(data_size))
	max_sent_len = 0
	max_num_sent = 0
	for i in range(batch_size):
		sentences = list(splitOnDelim( data[shuffle_indices[i]],[0])) # Add indices of delimiters
		if(len(sentences) > max_num_sent):
			max_num_sent = len(sentences)
		max_len = max(len(x) for x in sentences)
		if(max_len > max_sent_len):
			max_sent_len = max_len 

	batch = np.full((batch_size,max_num_sent,max_sent_len),-1)

	print max_num_sent
	print max_sent_len

	for i in range(batch_size):
		sentences = list(splitOnDelim(data[shuffle_indices[i]],[0]))
		for j in range(len(sentences)):
			for k in range(len(sentences[j])):
				batch[i,j,k] = sentences[j][k]

	return batch


"""
Split context into list of sentences. Pass the indices of sentence ending markers in the vocab as delimiter

delim : list of delimiters
"""
def splitOnDelim(seq, delim):
	s = []
	for w in seq:
		s.append(w)
		if w in delim:
			yield s 
			s = []		
	yield s


"""
def batch_iter(data, num_epochs=30, batch_size=32, shuffle=True):
    """
#    Purpose:
 #       Generates a batch iterator for a dataset.
  #  Credits: https://github.com/dennybritz/cnn-text-classification-tf
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
"""





