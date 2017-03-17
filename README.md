# sentence-selection
## Sentence Selection for Reading Comprehension task on the SQuaD question answering dataset
Reading Comprehension is the task of having the reader answer questions based on the given piece of text.

While most reading comprehension models currently are trained end-to-end, this task can be split into two disctinct parts: 
1. identifying sentences in the passage that are relevant to the question and 
2. extracting the answer from the relevant sentences. 
This model focuses on part 1 of this reading comprehension task; 
moreover, it focuses on predicting which one sentence in the context passage contains the correct answer to the question.

The Stanford Question Answering Dataset(https://rajpurkar.github.io/SQuAD-explorer/) is used for experimentation.
The dataset has the unique property of having word spans of the original text passage as answers rather than single word or multiple choice answers. Since the overwhelming majority of answers to SQuAD questions are contained within one sentence, we have a gold label for which sentence in the passage had the answer to the question.

The model creates vector representations for each question and context sentence. We then used a similarity metric between each sentence vector and the corresponding question vector to score the â€relevanceâ€ of each sentence in the paragraph to the question. The sentence and question vector representations are created by concatenating the final hidden state vectors after running a bidirectional Gated Recurrent Unit RNN (Cho et al., 2014) over the word embedding vectors.

### Training the model 
The model has been run in Tensorflow v0.11 . 
Run the file model_train.py to train the model. The hyperparameters for training the model can be set in the model_train.py file. 
The preprocessed training and dev data files are available in the data folder. The code for preprocessing the data is in data_utils.py file.
