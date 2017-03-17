import tensorflow as tf
import numpy as np
from rnn_cell import GRUCell
from rnn import bidirectional_rnn, rnn, rnn_decoder_attention
from attention import BilinearFunction

class AttentionSentenceClassifier(object):
    def __init__(self, hidden_size=128, vocab_size=42791, embedding_size=50, embedding_matrix=None, \
        embedding_trainable=False, sentence_rep_str="cbow", question_rep_str="cbow"):
        tf.set_random_seed(1234)

        # Placeholders
        # ==================================================
        # (batch_size * max_sentence_count x max_sentence_length)
        self.sentences = tf.placeholder(tf.int32, [None, None], name="sentences")
        self.questions = tf.placeholder(tf.int32, [None, None], name="questions")
        self.labels = tf.placeholder(tf.int32, [None, ], name="labels")

        attending_size = hidden_size

        # Input Preparation (Mask Creation)
        # ==================================================
        with tf.variable_scope("masks"):
            # MASK SENTENCES
            # (batch_size * mask_sentence_count x max_sentence_length)
            sentence_mask = tf.cast(self.sentences >= 0, tf.int32)
            #sentence_mask = tf.sequence_mask(doc_lengths, dtype=tf.int32)
            masked_sentences = tf.mul(self.sentences, sentence_mask)

            batch_size = tf.shape(self.questions)[0]

            # (batch_size x max_sent_per_doc)
            batch_mask = tf.reshape(tf.reduce_max(sentence_mask, 1), [batch_size, -1])
            answer_counts = tf.cast(tf.reduce_sum(batch_mask, 1), tf.float32)

            # MASK QUESTIONS
            # create mask (batch_size x max_question_length)
            question_mask = tf.cast(self.questions >= 0, tf.int32)
            masked_question = tf.mul(question_mask, self.questions)
            question_mask_float = tf.expand_dims(tf.cast(question_mask, tf.float32), -1)

            max_sent_per_doc = tf.cast(tf.shape(sentence_mask)[0]/batch_size, tf.int32)
            max_sent_len = tf.shape(self.sentences)[1]
            max_ques_len = tf.shape(self.questions)[1]

            document_mask = tf.cast(tf.reshape(sentence_mask, [batch_size, max_sent_per_doc * max_sent_len]), tf.float32)

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

            masked_documents = tf.reshape(masked_sentences, [batch_size, max_sent_per_doc * max_sent_len])
            document_embeddings = tf.gather(self.W_embeddings, masked_documents)
            masked_document_embeddings = tf.mul(document_embeddings, document_mask)

            # (batch_size x max_question_length x embedding_size)
            question_embeddings = tf.gather(self.W_embeddings, masked_question)
            masked_question_embeddings = tf.mul(question_embeddings, question_mask_float)

        # Query Representation (CBOW or RNN)
        # ==================================================
        with tf.variable_scope("query-representation"):

            # RNN Attention on sentence embeddings -----------------------------------------
            doc_sentences = masked_document_embeddings

            question_cbow = tf.reduce_mean(masked_question_embeddings, 1)

            self.decoder_cell = GRUCell(state_size=hidden_size, input_size=embedding_size*2, scope="GRU_decoder")
            self.bilinearf = BilinearFunction(attending_size=hidden_size, attended_size=embedding_size)

            hidden_states_decoder, last_state_decoder, alpha_weights_time = \
                rnn_decoder_attention(cell=self.decoder_cell,
                                            start_state=question_cbow, # ZEROS
                                            inputs=question_embeddings,
                                            inputs_mask=tf.cast(question_mask, tf.float32),
                                            attentionf=self.bilinearf,
                                            attended=document_embeddings, #doc_sentences,
                                            attended_mask=tf.cast(document_mask, tf.float32)
                                          )
            # alpha_weights originally (batch x context_time x max_ques_len)
            alpha_weights_time_prime = tf.reshape(alpha_weights_time, [batch_size, max_sent_len, max_sent_per_doc, max_ques_len]) #
            # (batch_size, max_sent_per_doc, max_ques_len)
            alpha_weights_per_qword = tf.reduce_sum(alpha_weights_time_prime, 1)
            # (batch_size, max_sent_per_doc)
            sent_score = tf.reduce_sum(alpha_weights_per_qword, 2)

            # normalize
            self.probabilities = tf.div(sent_score, tf.reduce_sum(sent_score))

        # Loss
        # ==================================================
        with tf.variable_scope("prediction"):

            one_hot_labels = tf.one_hot(self.labels, max_sent_per_doc, dtype=tf.float32)

            likelihoods = tf.reduce_sum(tf.mul(self.probabilities, one_hot_labels), 1)
            likelihoods = tf.div(likelihoods, answer_counts)
            log_likelihoods = tf.log(likelihoods+0.00000000000000000001)
            self.loss = tf.div(tf.mul(tf.reduce_sum(log_likelihoods), -1), tf.cast(batch_size, tf.float32))
            correct_vector = tf.cast(tf.equal(self.labels, tf.cast(tf.argmax(self.probabilities, 1), tf.int32)), \
                tf.float32, name="correct_vector")
            self.accuracy = tf.reduce_mean(correct_vector)
