This project tagges parts of speech text, using NLP techniques:
1) features extraction out of training text, in a format of "word_tag" for all words in text.
2) finding the best weights for those features, using gradient descent.
3) inference over a new unseen text, using Viterbi algorithm (dinamic programming).


in the file "interfaces.py" you can use our model in order to:
1) train the model
2) test the accuracy comparing to another Ground-Truth model
3) check the accuracy on the training data itself, using cross validation
4) tag a new unseen text file
