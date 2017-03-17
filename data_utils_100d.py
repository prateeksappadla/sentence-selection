# -*- coding: utf-8 -*-
"""
Structure of Json File

-data : list (One data object corresponds to one topic in Wikipedia. Paragraph is one paragraph from the article)
	-[paragraphs]
		- context
		- qas
			-[question]
				-id
				-[answers]
					-text
					-answer_start
"""

import numpy as np
import json
from nltk import word_tokenize, sent_tokenize
import re
import itertools
import collections
import pickle

###########################################################################################
def digit_to_N(word):
    # 3.3 => n; 3,33.3 =>nnn
    if re.match(r'^(\d+([.]|[,]))*\d+$', word):
        word = re.sub(r'[,]', '', word)
        integer_len = len(re.match(r'^\d+', word).group())
        return 'n'*integer_len if integer_len <= 4 else 'nnnn'
    # 333333 => nnnn
    elif re.match(r'^\d+$', word):
        word_len = len(word)
        return 'n'*word_len if word_len <= 4 else 'nnnn'
    # 3,333 => NNNN
    # elif re.match(r'^(\d+[,])*\d+$', word):
    #     word_len = len(re.sub(r'[,]', '', word))
    #     return 'N' * word_len if word_len <= 5 else 'NNNNN'
    else:
        return word


def wordlist_corner_case_tokenize(wordlist):
    return list(itertools.chain.from_iterable([word_corner_case_tokenize(word) for word in wordlist]))

def seperate_dash_colon(word):
    # re.match(r'^(.*[-].*)\1*')
    return re.sub(r'[:]', ' : ', re.sub(r'-|—|–|_', ' - ', word))


def seperate_slash_backslash(word):
    return re.sub(r'[/]', ' / ', re.sub(r'[\\]', ' \\ ', word))


def word_corner_case_tokenize(word):
    word = seperate_dash_colon(word)
    word = seperate_slash_backslash(word)
    return [digit_to_N(w) for w in word.split()]

#################################################################################################

with open('dev-v1.1.json') as data_file:
	raw_data = json.load(data_file)

"""
Input: A sentence
Output: List of words(tokens)
"""
def tokenize(sentence):
	words = [s.lower() for s in word_tokenize(sentence)]
	return wordlist_corner_case_tokenize(words)

"""
Input: Dataset
Output:
"""
def tokenize_data():
	ind = 0
	qid = 0
	context_dict = {}
	question_dict = {}
	answer_dict = {}
	words = []
	for article in raw_data["data"]:
		for paragraph in article["paragraphs"]:
			sentence_lens = []
			tokenized_context = []
			sentences = sent_tokenize(paragraph["context"])
			for sentence in sentences:
				sentence_lens.append(len(sentence))
				sent_words = tokenize(sentence)
				words.extend(sent_words)
				tokenized_context.append(sent_words)
			context_dict[ind]= tokenized_context
			for questionId in paragraph["qas"]:
				question = questionId["question"]
				question_words = tokenize(question)
				words.extend(question_words)
				question_dict[qid] = [question_words, ind]
				answer = questionId["answers"][0]
				answer_start = answer["answer_start"]
				char_sum = 0
				for i,slen in enumerate(sentence_lens):
					char_sum += slen
					if answer_start < char_sum:
						answer_dict[qid] = i
						break
					char_sum += 1
				qid += 1
			ind += 1
	return context_dict, question_dict, answer_dict,words


# TODO: create dictionary by taking intersection of vocabulary of Glove and Squad
def buildSquadVocabDict(words):
	word_counts = collections.Counter(words)
	vocab = list(word_counts)
	#word_to_count_dict = dict(word_counts)
	vocab_dict = {}
	ind  = 0
	for word in vocab:
		vocab_dict[word] = ind
		ind += 1
	return vocab_dict

def encodeContextToIndices(dictionary, context_dict):
	encoded_context_dict = {}
	for key in context_dict.keys():
		context = context_dict[key]
		for sentence in context:
			for i,word in enumerate(sentence):
				if(dictionary.get(word) == None):
					sentence[i] = dictionary.get('unk')
				else:
					sentence[i] = dictionary.get(word)
		encoded_context_dict[key] = context
	return encoded_context_dict

def encodeQuestionToIndices(dictionary, question_dict):
	encoded_question_dict = {}
	for key in question_dict.keys():
		question = question_dict[key]
		question_text = question[0]
		for i,word in enumerate(question_text):
			if(dictionary.get(word) == None):
				question_text[i] = dictionary.get('unk')
			else:
				question_text[i] = dictionary.get(word)
		question[0] = question_text
		encoded_question_dict[key] = question
	return encoded_question_dict


# Batch Iterator

def batch_iter(context_dict, question_dict, answer_dict, num_epochs=30, batch_size=32):
    """
	Purpose: Generates a batch iterator for a dataset.
  	Credits: https://github.com/dennybritz/cnn-text-classification-tf
    """
    qlist = question_dict.values();
    alist = answer_dict.values();
    assert len(qlist) == len(alist)
    data_size = len(qlist)
    num_batches_per_epoch = int(data_size/batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_qlist = [qlist[ind] for ind in shuffle_indices]
        shuffled_alist = [alist[ind] for ind in shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            qbatch = shuffled_qlist[start_index:end_index]
            abatch = shuffled_alist[start_index:end_index]
            contexts, questions, answers = prepare_batch(context_dict, qbatch, abatch, batch_size)
            yield contexts, questions, answers


def prepare_batch(context_dict, qbatch, abatch, batch_size=32):
	cbatch = []
	for q in qbatch:
		cbatch.append(context_dict[q[1]])

	max_num_sent = max(len(c) for c in cbatch)

	sentences = []
	for c in cbatch:
		for s in c:
			sentences.append(s)
	max_sent_len = max(len(s) for s in sentences)

	max_q_len = max(len(q[0]) for q in qbatch)

	context_batch = np.full((batch_size * max_num_sent, max_sent_len ),-1,dtype=np.int32)
	question_batch = np.full((batch_size, max_q_len), -1,dtype=np.int32 );
	answer_batch = np.array(abatch)

	for i in range(len(qbatch)):
		for j in range(len(cbatch[i])):
			for k in range(len(cbatch[i][j])):
				context_batch[i * max_num_sent + j,k] = cbatch[i][j][k]
		for j in range(len(qbatch[i][0])):
			question_batch[i,j] = qbatch[i][0][j]

	return context_batch, question_batch, answer_batch


cdict,qdict,adict,words = tokenize_data()

with open('glovedict_json_100d.json', 'r') as dict_json_file:
	vocab_dict = json.load(dict_json_file)

encoded_cdict = encodeContextToIndices(vocab_dict,cdict)
encoded_qdict = encodeQuestionToIndices(vocab_dict,qdict)

with open('dev_context_dict_100d.p','wb') as cdict_pickle_file:
	pickle.dump(encoded_cdict,cdict_pickle_file)

with open('dev_question_dict_100d.p','wb') as qdict_pickle_file:
	pickle.dump(encoded_qdict,qdict_pickle_file)

with open('dev_answer_dict_100d.p','wb') as adict_pickle_file:
	pickle.dump(adict,adict_pickle_file)

# for contexts,questions,answers in batch_iter(encoded_cdict,encoded_qdict,adict,1,6):
# 	print contexts
# 	print questions
# 	print answers
