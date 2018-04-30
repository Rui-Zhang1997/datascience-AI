from nltk import word_tokenize, sent_tokenize
import numpy as np

clean_text = lambda text: [[w for w in word_tokenize(sentence.lower()) if w.isalnum()] for sentence in sent_tokenize(text)]

def generate_dictionary(sentences, sent_tokenized=False):
    wordset = set()
    for sent in sentences:
        wordset.update(sent)
    vocabulary = {}
    reversed_vocabulary = {}
    for i, word in enumerate(wordset):
        vocabulary[i] = word
        reversed_vocabulary[word] = i
    return vocabulary, reversed_vocabulary

divide_set = lambda sentences, ratio=0.7: (sentences[:int(len(sentences) * ratio)], sentences[int(len(sentences) * ratio):])
indexize_words = lambda words, dictionary: [dictionary[w] for w in words]

def flatten(lst):
    res = []
    for l in lst:
        if isinstance(l, list):
            res += flatten(res)
        else:
            res.append(l)
    return res

indexize_wordsets = lambda wordsets, dictionary: [indexize_words(wordset, dictionary) for wordset in wordsets]
class LSTMBatchGenerator:
    def __init__(self, data, num_steps, batch_size, vocabulary, skip_steps=1):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.vocabulary = vocabulary
        self.idx = 0
        self.skip_steps = skip_steps

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, self.vocabulary))
        while True:
            for i in range(self.batch_size):
                if self.idx + self.num_steps >= len(self.data):
                    self.idx = 0
                x[i] = self.data[self.idx:self.idx + self.num_steps]
                tmp_y = self.data[self.idx + 1: self.idx + self.num_steps + 1]
                y[i] = to_categorical(tmp_y, num_classes=self.vocabulary)
                self.idx += self.skip_step
            yield x, y
    
    @classmethod
    def create(self, data, cleaned=False):
        cleaned_text = clean_text(data) if not cleaned else data
        dictionary, reverse_dictionary = generate_dictionary(cleaned_text)
        index_sentences = indexize_wordsets(cleaned_text, reverse_dictionary)
        train_set, validate_set = divide_set(index_sentences)
        return LSTMBatchGenerator(flatten(train_set), 5, 5, len(dictionary)), LSTMBatchGenerator(flatten(validate_set), 5, 5, len(dictionary))

sample = """
In the first couple of lines our x and y output arrays are created. The size of variable x is fairly straight forward to understand – it’s first dimension is the number of samples we specify in the batch. The second dimension is the number of words we are going to base our predictions on. The size of variable y is a little more complicated. First it has the batch size as the first dimension, then it has the number of time steps as the second, as discussed above. However, y has an additional third dimension, equal to the size of our vocabulary, in this case 10,000.

The reason for this is that the output layer of our Keras LSTM network will be a standard softmax layer, which will assign a probability to each of the 10,000 possible words. The one word with the highest probability will be the predicted word – in other words, the Keras LSTM network will predict one word out of 10,000 possible categories. Therefore, in order to train this network, we need to create a training sample for each word that has a 1 in the location of the true word, and zeros in all the other 9,999 locations. It will look something like this: (0, 0, 0, …, 1, 0, …, 0, 0) – this is called a one-hot representation, or alternatively, a categorical representation. Therefore, for each target word, there needs to be a 10,000 length vector with only one of the elements in this vector set to 1.

Ok, now onto the while True: yield x, y paradigm that was discussed earlier for the generator. In the first line, we enter into a for loop of size batch_size, to populate all the data in the batch. Next, there is a condition to test regarding whether we need to reset the current_idx pointer. Remember that for each training sample we consume num_steps words. Therefore, if the current index point plus num_steps is greater than the length of the data set, then the current_idx pointer needs to be reset to zero to start over with the data set.

After this check is performed, the input data is consumed into the x array. The data indices consumed is pretty straight-forward to understand – it is the current index to the current-index-plus-num_steps number of words. Next, a temporary y variable is populated which works in pretty much the same way – the only difference is that the starting point and the end point of the data consumption is advanced by 1 (i.e. + 1). If this is confusing, please refer to the “cat sat on the mat etc.” example discussed above.
"""

trainBatch, validBatch = LSTMBatchGenerator.create(sample)
for i in trainBatch.generate():
    print(i)
