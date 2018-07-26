# Online-Writer-Identification

In this project an end-to-end framework is proposed for online text-independent writer identification by using a recurrent neural network. Specifically, the handwriting data of a particular writer are represented by a set of random hybrid strokes (RHSs). Each RHS is a randomly sampled short sequence representing pen tip movements (xy-coordinates) and pen-down or pen-up states. RHS is independent of the content and language involved in handwriting; therefore, writer identification at the RHS level is more general and convenient than the character level or the word level, which also requires character/word segmentation. The RNN model with bidirectional long short-term memory is used to encode each RHS into a fixed-length vector for final classification. All the RHSs of a writer are classified independently, and then, the posterior probabilities are averaged to make the final decision. The proposed framework is end-to-end and does not require any domain knowledge for handwriting data analysis. Experiments on both English (133 writers) and Chinese (186 writers) databases verify the advantages of the method compared with other state-of-the-art approaches. Experiments on both English and Chinese databases resulted in >95% accuracy.

Framework Used
-
The project was completely done under Torch framework using luarocks modules:
nn, dpnn, rnn, optim and nninit

Data
-
 The used dataset is the handwriting database from the BIT: http://biometrics.idealtest.org/dbDetailForUser.do?id=10
 The database consists of two datasets, the first dataset is written in Chinese by 187 writers, while the second dataset is      written in English by 134 writers. 
 
Files Description
 -
 data_preprocess:
 script to convert the data into a text file consisting of xy-coordinates and pen-down or pen-up states at each instance of all writers arranged in alphabetical order. 
 
 data.lua:
 file to load training and testing data as well as labels from the text file to torch.CudaTensors
 
 main.lua: 
file in which the BiRNN architecture and the evaluating loss function is defined. The weights and biases of the BiRNN model are initialised and the model is then trained on the training data. Theweights and biases of the trained model are saved and the model is evaluated on the testing data.
 
 train.lua:
 file which loads pre-trained models and saves new weights and biases after further training.
 
 eval.lua:
 file to evaluate trained models on testing data.
 
Futher Description of the project:
 -
https://docs.google.com/document/d/1ml8o443tkUdLuPw8f9LtL5AAbIA--VjhbUd4Byh4W5Y/edit?usp=sharing
