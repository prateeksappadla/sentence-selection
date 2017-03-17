import tensorflow as tf
import numpy as np
from rnn_cell import GRUCell
from rnn import bidirectional_rnn, rnn, rnn_decoder_attention
from attention import BilinearFunction

class SentenceClassifier(object):
    def __init__(self, hidden_size=128, vocab_size=65011, embedding_size=50, embedding_matrix=None, \
        embedding_trainable=False, sentence_rep_str="cbow", question_rep_str="cbow"):
        tf.set_random_seed(1234)

        # Placeholders
        # ==================================================
        # (batch_size * max_sentence_count x max_sentence_length)
        self.sentences = tf.placeholder(tf.int32, [None, None], name="sentences")
        self.questions = tf.placeholder(tf.int32, [None, None], name="questions")
        self.labels = tf.placeholder(tf.int32, [None, ], name="labels")

        # initialize dimension variables based on contructor arguments
        if sentence_rep_str == "cbow":
            attended_size = embedding_size
        elif sentence_rep_str == "rnn":
            attended_size = hidden_size*2
        else:
            raise ValueError("Invalid `sentence_rep_str` argument; choose 'cbow' or 'rnn'.")

        if question_rep_str == "cbow":
            attending_size = embedding_size
        elif question_rep_str == "rnn":
            attending_size = hidden_size*2
        else:
            raise ValueError("Invalid `question_rep_str` argument; choose 'cbow', 'rnn', or 'rnn-attention'.")

        # Input Preparation (Mask Creation)
        # ==================================================
        with tf.variable_scope("masks"):
            # MASK SENTENCES
            # (batch_size * mask_sentence_count x max_sentence_length)
            sentence_mask = tf.cast(self.sentences >= 0, tf.int32)
            #sentence_mask = tf.sequence_mask(doc_lengths, dtype=tf.int32)
            masked_sentences = tf.mul(self.sentences, sentence_mask)

            batch_size = tf.shape(self.questions)[0]

            # RESHAPE SENTENCE MASK
            # (batch_size x max_sent_per_doc)
            batch_mask = tf.reshape(tf.reduce_max(sentence_mask, 1), [batch_size, -1])
            answer_counts = tf.cast(tf.reduce_sum(batch_mask, 1), tf.float32)
            # (batch_size * max_sent_per_doc x 1 x 1)
            sentence_batch_mask = tf.cast(tf.reshape(batch_mask, [-1, 1, 1]), tf.float32)

            # MASK QUESTIONS
            # create mask (batch_size x max_question_length)
            question_mask = tf.cast(self.questions >= 0, tf.int32)
            masked_question = tf.mul(question_mask, self.questions)
            question_mask_float = tf.expand_dims(tf.cast(question_mask, tf.float32), -1)

            max_sent_per_doc = tf.cast(tf.shape(sentence_mask)[0]/batch_size, tf.int32)

        # Embeddings
        # ==================================================
        with tf.variable_scope("embeddings"):
            if embedding_matrix is None:
                self.W_embeddings = tf.get_variable(shape=[vocab_size, embedding_size], \
                                               initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                               name="W_embeddings", dtype=tf.float32)
            else:
                ################## option to use pre-trained embeddings ##################
                self.W_embeddings = tf.Variable(embedding_matrix, \
                                               name="W_embeddings", dtype=tf.float32, trainable=embedding_trainable)

            # batch_size * max_sent_per_doc x max_sentence_length x embedding_size
            sentence_embeddings = tf.gather(self.W_embeddings, masked_sentences)
            masked_sentence_embeddings = tf.mul(sentence_embeddings, tf.cast(tf.expand_dims(sentence_mask, -1), tf.float32))

            # (batch_size x max_question_length x embedding_size)
            question_embeddings = tf.gather(self.W_embeddings, masked_question)
            masked_question_embeddings = tf.mul(question_embeddings, question_mask_float)

        # Sentence Representation (CBOW or RNN)
        # ==================================================
        with tf.variable_scope("sentence-representation"):

            # CBOW -----------------------------------------
            if sentence_rep_str == "cbow":
                # (batch_size * max_sentence_count x embedding_size)
                cbow_sentences = tf.reduce_mean(masked_sentence_embeddings, 1)
                # reshape batch to (batch_size x max_doc_length x embedding_size)
                doc_sentences = tf.reshape(cbow_sentences, [batch_size, -1, embedding_size])

                self.sentence_representation = cbow_sentences

            # RNN -----------------------------------------
            elif sentence_rep_str == "rnn":
                self.forward_cell_d = GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU-Forward-D")
                self.backward_cell_d = GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU-Backward-D")

                self.hidden_states_d, last_state_d = bidirectional_rnn(self.forward_cell_d, self.backward_cell_d, \
                    sentence_embeddings, tf.cast(sentence_mask, tf.float32), concatenate=True)

                doc_sentences = tf.reshape(last_state_d, [batch_size, -1, hidden_size*2])

                self.sentence_representation = last_state_d

        # Query Representation (CBOW or RNN)
        # ==================================================
        with tf.variable_scope("query-representation"):

            # CBOW -----------------------------------------
            if question_rep_str == "cbow":
                # (batch_size x embedding_size)
                question_cbow = tf.reduce_mean(masked_question_embeddings, 1)
                self.question_representation = question_cbow

            # RNN -----------------------------------------
            elif question_rep_str == "rnn":
                self.forward_cell_q = GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU-Forward-Q")
                self.backward_cell_q = GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU-Backward-Q")

                self.hidden_states_q, last_state_q = bidirectional_rnn(self.forward_cell_q, self.backward_cell_q, \
                    question_embeddings, tf.cast(question_mask, tf.float32), concatenate=True)

                #tf.reduce_mean(self.hidden_states_q, )

                self.question_representation = last_state_q


        # Similarity Scoring
        # ==================================================
        # Using simple dot product/cosine similiarity as of now (https://arxiv.org/pdf/1605.07427v1.pdf)

        with tf.variable_scope("similarity-scoring"):

            # (batch_size x max_sent_per_doc)
            attention = BilinearFunction(attending_size=attending_size, attended_size=attended_size)
            alpha_weights, attend_result = attention(self.question_representation, attended=doc_sentences, \
                 time_mask=tf.cast(batch_mask, tf.float32))
            self.probabilities = alpha_weights

        # Loss
        # ==================================================
        with tf.variable_scope("prediction"):

            one_hot_labels = tf.one_hot(self.labels, max_sent_per_doc, dtype=tf.float32)

            likelihoods = tf.reduce_sum(tf.mul(self.probabilities, one_hot_labels), 1)
            likelihoods = tf.div(likelihoods, answer_counts)
            log_likelihoods = tf.log(likelihoods+0.00000000000000000001)
            self.loss = tf.div(tf.mul(tf.reduce_sum(log_likelihoods), -1), tf.cast(batch_size, tf.float32))
            self.correct_vector = tf.cast(tf.equal(self.labels, tf.cast(tf.argmax(self.probabilities, 1), tf.int32)), tf.float64, name="correct_vector")
            self.predict_labels = tf.argmax(self.probabilities, 1)
            self.accuracy = tf.reduce_mean(self.correct_vector)
