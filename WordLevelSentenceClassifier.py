import tensorflow as tf
import numpy as np
from rnn_cell import GRUCell
from rnn import bidirectional_rnn, rnn
from attention import BilinearFunction

def getFLAGS():
    # Model Hyperparameters
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_integer("num_nodes", 16, "Number of nodes in fully connected layer")
    tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "Weight lambda on l2 regularization")

    # Training Parameters
    tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
    tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("patience", 800, "Minimum number of batches seen before early stopping")
    tf.flags.DEFINE_integer("patience_increase", 6, "Number of dev evaluations of increasing loss before early stopping")

    # Display/Saving Parameters
    tf.flags.DEFINE_integer("evaluate_every", 10, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

    # Print
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    return FLAGS

class WordLevelSentenceClassifier(object):
    def __init__(self, hidden_size=10, vocab_size=42791, embedding_size=50, embedding_matrix=None):
        #tf.set_random_seed(1234)

        # Placeholders
        # ==================================================
        # (batch_size * max_sentence_count x max_sentence_length)
        self.sentences = tf.placeholder(tf.int32, [None, None], name="sentences")
        self.questions = tf.placeholder(tf.int32, [None, None], name="questions")
        self.labels = tf.placeholder(tf.int32, [None, ], name="labels")

        batch_size = tf.shape(self.questions)[0]
        #self.batch_size = tf.placeholder(tf.int32, name="batch_size")
        sentences = self.sentences
        questions = self.questions
        labels = self.labels

        # (batch_size * mask_sentence_count x max_sentence_length)
        sentence_mask = tf.cast(sentences >= 0, tf.int32)
        self.sentence_mask = sentence_mask
        #sentence_mask = tf.sequence_mask(doc_lengths, dtype=tf.int32)
        masked_sentences = tf.mul(sentences, sentence_mask)

        max_sent_per_doc = tf.cast(tf.shape(sentence_mask)[0]/batch_size, tf.int32)
        self.max_sent_per_doc = max_sent_per_doc

        # Input Preparation
        # ==================================================
        with tf.variable_scope("embeddings"):
            if embedding_matrix == None:
                self.W_embeddings = tf.get_variable(shape=[vocab_size, embedding_size], \
                                               initializer=tf.random_uniform_initializer(-0.01, 0.01),\
                                               name="W_embeddings", dtype=tf.float32)
            else:
                ################## option to use pre-trained embeddings ##################
                self.W_embeddings = tf.Variable(embedding_matrix, \
                                               name="W_embeddings", dtype=tf.float32)

            # SENTENCES MASKED
            # (batch_size x max_sent_per_doc)
            batch_mask = tf.reshape(tf.reduce_max(sentence_mask, 1), [batch_size, -1])
            # (batch_size * max_sent_per_doc x 1 x 1)
            sentence_batch_mask = tf.cast(tf.reshape(batch_mask, [-1, 1, 1]), tf.float32)

            # batch_size * max_sent_per_doc x max_sentence_length x embedding_size
            sentence_embeddings = tf.gather(self.W_embeddings, masked_sentences)
            masked_sentence_embeddings = tf.mul(sentence_embeddings, tf.cast(tf.expand_dims(sentence_mask, -1), tf.float32))

            # QUERY MASKED
            # create mask (batch_size x max_question_length)
            question_mask = tf.cast(questions > 0, tf.int32)
            self.question_mask = question_mask
            masked_question = tf.mul(question_mask, questions)
            self.masked_question = masked_question

            # (batch_size x max_question_length x embedding_size)
            question_embeddings = tf.gather(self.W_embeddings, masked_question)
            self.question_embeddings = question_embeddings
            question_mask_float = tf.expand_dims(tf.cast(question_mask, tf.float32), -1)
            masked_question_embeddings = tf.mul(question_embeddings, question_mask_float)
            self.masked_question_embeddings = masked_question_embeddings


        # CBOW Sentence Representation
        # ==================================================
        with tf.variable_scope("sentence-representation"):

            self.forward_cell_d = GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU-Forward-D")
            self.backward_cell_d = GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU-Backward-D")

            self.hidden_states_d, last_state_d = bidirectional_rnn(self.forward_cell_d, self.backward_cell_d, \
                sentence_embeddings, tf.cast(sentence_mask, tf.float32), concatenate=True)

            doc_sentences = tf.reshape(last_state_d, [batch_size, -1, hidden_size*2])
            self.cbow_sentences = doc_sentences

            # # (batch_size * max_sentence_count x embedding_size)
            # cbow_sentences = tf.reduce_mean(masked_sentence_embeddings, 1)
            # self.cbow_sentences = cbow_sentences
            # # reshape batch to (batch_size x max_doc_length x embedding_size)
            # doc_sentences = tf.reshape(cbow_sentences, [batch_size, -1, embedding_size])

        # Query Representation
        # ==================================================
        with tf.variable_scope("query-representation"):
            # easy baseline: cbow
            # (batch_size x embedding_size)

            #question_cbow = tf.reduce_mean(masked_question_embeddings, 1)
            #self.question_cbow = question_cbow

            self.forward_cell_q = GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU-Forward-Q")
            self.backward_cell_q = GRUCell(state_size=hidden_size, input_size=embedding_size, scope="GRU-Backward-Q")

            self.hidden_states_q, last_state_q = bidirectional_rnn(self.forward_cell_q, self.backward_cell_q, \
                question_embeddings, tf.cast(question_mask, tf.float32), concatenate=True)

            self.question_cbow = last_state_q

            # can use RNN representation as well*************************************

        # Similarity Scoring
        # ==================================================
        # Using simple dot product/cosine similiarity as of now (https://arxiv.org/pdf/1605.07427v1.pdf)

        with tf.variable_scope("similarity-scoring"):
            # (batch_size x max_sent_per_doc)
            #"""
            # dot_prod = tf.squeeze(tf.batch_matmul(doc_sentences, tf.expand_dims(question_cbow, -1)), [-1])
            # self.dot_prod = dot_prod
            #
            # # softmax
            # numerator = tf.exp(tf.sub(dot_prod, tf.expand_dims(tf.reduce_max(dot_prod, 1), -1))) * tf.cast(batch_mask, tf.float32)
            # denom = tf.reduce_sum(numerator, 1)
            #
            # # Dimensions (batch x time)
            # probabilities = tf.div(numerator, tf.expand_dims(denom, 1))
            #"""

            # #(batch_size x max_sent_per_doc)
            # sentence_norm = tf.sqrt(tf.reduce_sum(tf.mul(doc_sentences, doc_sentences), -1))
            # self.sentence_norm = sentence_norm
            # # (batch_size)
            # question_norm = tf.sqrt(tf.reduce_sum(tf.mul(question_cbow, question_cbow), 1))
            # self.question_norm = question_norm
            #
            # denom = tf.mul(sentence_norm, tf.expand_dims(question_norm, -1))+1e-30
            # self.denom = denom
            # # (batch_size x max_sent_per_doc) - scalars between -1 and +1
            # cosine_similarity = tf.div(dot_prod, denom)
            # self.cosine_similarity = cosine_similarity
            #
            # masked_pos_cos_sim = tf.sub(tf.add(cosine_similarity, 1), tf.cast(batch_mask < 1, tf.float32))
            # self.masked_pos_cos_sim = masked_pos_cos_sim
            # normalized_cos_sim = tf.div(masked_pos_cos_sim, tf.expand_dims(tf.reduce_sum(masked_pos_cos_sim, 1), -1))

            #"""
            attention = BilinearFunction(attending_size=hidden_size*2, attended_size=hidden_size*2)
            alpha_weights, attend_result = attention(self.question_cbow, attended=doc_sentences, \
                time_mask=tf.cast(batch_mask, tf.float32))
            probabilities = alpha_weights

            #"""

            #probabilities = tf.abs(dot_prod) #normalized_cos_sim
            self.probabilities = probabilities

        with tf.variable_scope("prediction"):
            one_hot_labels = tf.one_hot(labels, max_sent_per_doc, dtype=tf.float32)

            likelihoods = tf.reduce_sum(tf.mul(probabilities, one_hot_labels), 1)
            log_likelihoods = tf.log(likelihoods+0.00000000000000000001)
            self.loss = tf.mul(tf.reduce_sum(log_likelihoods), -1)
            correct_vector = tf.cast(tf.equal(labels, tf.cast(tf.argmax(probabilities, 1), tf.int32)), \
                tf.float32, name="correct_vector")
            self.accuracy = tf.reduce_mean(correct_vector)
