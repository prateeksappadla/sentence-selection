#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from SentenceClassifier import SentenceClassifier
from AttentionSentenceClassifier import AttentionSentenceClassifier
from tensorflow.contrib import learn
import data_utils
import pickle
import json

# ======================== MODEL HYPERPARAMETERS ========================================
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "Weight lambda on l2 regularization")

# Training Parameters
tf.flags.DEFINE_integer("hidden_size", 128, "RNN hidden size")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("max_global_norm", 5, "Max gradient norm to warrant clipping")
tf.flags.DEFINE_boolean("embedding_trainable", False, "Backpropagate into pretrained embedding matrix boolean")
tf.flags.DEFINE_string("sentence_rep_str", "cbow", "Sentence representation model type to use") # cbow, rnn
tf.flags.DEFINE_string("question_rep_str", "rnn", "Question representation model type to use") # cbow, rnn

# Display/Saving Parameters
tf.flags.DEFINE_integer("evaluate_every", 2000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Print
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# =============================== PREPARING DATA FOR TRAINING/VALIDATION/TESTING ===============================================
print("Loading data...")

context_dict = pickle.load(open("context_dict_50d.p", "rb"))
question_dict = pickle.load(open("question_dict_50d.p", "rb"))
answer_dict = pickle.load(open("answer_dict_50d.p", "rb"))

dev_context_dict = pickle.load(open("dev_context_dict_50d.p", "rb"))
dev_question_dict = pickle.load(open("dev_question_dict_50d.p", "rb"))
dev_answer_dict = pickle.load(open("dev_answer_dict_50d.p", "rb"))

embedding_matrix = np.load("embedding_matrix_50d.npy")

# ================================================== MODEL TRAINING ======================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)

    with sess.as_default():

        sent_classifier = SentenceClassifier(hidden_size=FLAGS.hidden_size,
                                             vocab_size=65011,
                                             embedding_size=50,
                                             embedding_matrix=embedding_matrix,
                                             embedding_trainable=FLAGS.embedding_trainable,
                                             sentence_rep_str=FLAGS.sentence_rep_str,
                                             question_rep_str=FLAGS.question_rep_str
                                             )

        # sent_classifier = AttentionSentenceClassifier(hidden_size=FLAGS.hidden_size,
        #                                      vocab_size=42791,
        #                                      embedding_size=50,
        #                                      embedding_matrix=embedding_matrix,
        #                                      embedding_trainable=FLAGS.embedding_trainable,
        #                                      sentence_rep_str=FLAGS.sentence_rep_str,
        #                                      question_rep_str=FLAGS.question_rep_str
        #                                      )

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        trainables = tf.trainable_variables()
        grads = tf.gradients(sent_classifier.loss, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.max_global_norm)
        grads_and_vars = zip(grads, trainables)

        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Define Training procedure
        # global_step = tf.Variable(0, name="global_step", trainable=False)
        # optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
        # grads_and_vars = optimizer.compute_gradients(sent_classifier.loss)
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries_merged = None
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", sent_classifier.loss)
        acc_summary = tf.scalar_summary("accuracy", sent_classifier.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged]) if grad_summaries_merged \
            else tf.merge_summary([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def dev_step(context, question, answer, writer=None):

            # Evaluates model on a dev set
            feed_dict = {
                sent_classifier.sentences : context,
                sent_classifier.questions : question,
                sent_classifier.labels : answer
            }

            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, sent_classifier.loss, sent_classifier.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}\n\n".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
	

        # TODO=============================
        def error_analysis(dev_context_dict, dev_question_dict, dev_answer_dict):
            error_log = open('error_log_restore.txt', 'w')
            ##### dict to load #####
            word_to_index = json.load(open('glovedict_json_100d.json', 'r'))
            ##### dict to load #####
            index_to_word = {i: w for (w, i) in word_to_index.items()}
            batch_size = 100
            batches = data_utils.batch_iter(dev_context_dict, dev_question_dict,
                                                                             dev_answer_dict,
                                                                             num_epochs=1, batch_size=batch_size,
                                                                             shuffle=False)

            for batch_num, (c, q, a) in enumerate(batches):
                dev_contexts = c
                dev_questions = q
                dev_answers = a
                feed_dict = {
                    sent_classifier.sentences: dev_contexts,
                    sent_classifier.questions: dev_questions,
                    sent_classifier.labels: dev_answers
                }

                correct_vector, predict_labels = sess.run(
                    [sent_classifier.correct_vector, sent_classifier.predict_labels],
                    feed_dict)
                correct_vector = [bool(i) for i in list(correct_vector)]
                for correct_vector_index, is_correct in enumerate(correct_vector):
                    dict_index = batch_num * batch_size + correct_vector_index
                    if not is_correct:
                        print('===== context #%d =====' % dev_question_dict[dict_index][1], file=error_log)
                        print('context:', file=error_log)
                        print([[index_to_word[index] for index in sent]
                               for sent in dev_context_dict[dev_question_dict[dict_index][1]]], file=error_log)
                        print('question:', file=error_log)
                        print([index_to_word[index] for index in dev_question_dict[dict_index][0]], file=error_log)
                        print('gold answer:', file=error_log)
                        print([index_to_word[index] for index in
                               dev_context_dict[dev_question_dict[dict_index][1]][dev_answer_dict[dict_index]]], file=error_log)
                        print('predicted answer:', file=error_log)
                        if predict_labels[correct_vector_index] < len(dev_context_dict[dev_question_dict[dict_index][1]]):
                            print([index_to_word[index] for index in
                                   dev_context_dict[dev_question_dict[dict_index][1]][predict_labels[correct_vector_index]]], file=error_log)
                        else:
                            print('$$$$$ <PADDED SENTENCE> $$$$$', file=error_log)

        checkpoint_file = "/Users/prateek/Documents/qasystem/python/tensorflow/runs/1482094225/checkpoints/model-12000"
        saver.restore(sess, checkpoint_file)
        #print(sess.run(tf.all_variables()))	
        
        #final evaluation

        #error_analysis(dev_context_dict, dev_question_dict, dev_answer_dict)

        # dev_batches = data_utils.batch_iter(dev_context_dict, dev_question_dict, dev_answer_dict, num_epochs=1, batch_size=1000)
        # print("\nEvaluation:")
        # for dev_context, dev_question, dev_answer in dev_batches:
        #     dev_step(dev_context, dev_question, dev_answer, writer=dev_summary_writer)  


        dev_batches = data_utils.batch_iter(dev_context_dict, dev_question_dict, dev_answer_dict, num_epochs=1, batch_size=500)
        print("\nEvaluation:")
        num_batches = 0
        total_accuracy = 0.0
        for dev_context, dev_question, dev_answer in dev_batches:
            if(num_batches == 21):
                break
            feed_dict = {
                sent_classifier.sentences : dev_context,
                sent_classifier.questions : dev_question,
                sent_classifier.labels : dev_answer
            }
            [batch_accuracy] = sess.run([sent_classifier.accuracy],feed_dict)
            total_accuracy += batch_accuracy
            num_batches += 1
            print("Batch Accuracy: ",batch_accuracy)
        print("Accuracy: ",total_accuracy/21)        




       
