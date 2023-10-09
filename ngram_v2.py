import urllib.request
import re
import ssl
import math
import copy

# Define start, end, unknown-word tokens
start_token = "<STR>"  # Symbol to mark the start of a data point
end_token = "<STP>"    # Symbol to mark the end of a data point
unk_token = "<UNK>"

# URL for training data
train_url = "https://raw.githubusercontent.com/waseymulla/CS6320-HW1/main/train.txt"

# URL for test data
test_url = "https://raw.githubusercontent.com/waseymulla/CS6320-HW1/main/val.txt"

# Disable SSL certificate verification
context = ssl._create_unverified_context()

# Load and preprocess training data
train_file = urllib.request.urlopen(train_url, context=context)
unigram_counts = {}
bigram_counts = {}
previous_token = None

'''
# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\']', '', text)
    return text
'''

def preprocess_entry(text: str):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # remove special characters
    text = re.sub(r'(\s+)', ' ', text)          # remove multiple whitespaces
    return text

# Laplace (Add-one) Smoothing Function
def laplace_smoothing(word_count, total_token_count, vocab_size, smoothing_param=1):
    return (word_count + smoothing_param) / (total_token_count + smoothing_param * vocab_size)

# Add-k Smoothing Function
def add_k_smoothing(count, total_token_count, vocab_size, smoothing_param):
    return (count + smoothing_param) / (total_token_count + smoothing_param * vocab_size)

def evaluate_basic_model_unigram(corpus: list, token_counts: dict, total_tokens: int):
    validation_probs_log = []
    for review in corpus:
        log_review_prob = 0
        for token in review:
            # log since probability becomes very small on multiplication
            log_review_prob += math.log(token_counts.get(token, token_counts[unk_token])/total_tokens)
        validation_probs_log.append(log_review_prob)
    perplexity = math.exp(-sum(validation_probs_log)/total_tokens)
    return perplexity

def evaluate_basic_model_bigram(corpus: list, token_counts: dict, bigram_counts: dict, total_tokens: int):
    validation_probs_log = []
    for review in corpus:
        log_review_prob = 0
        for (word1, word2) in review:
            # process unknown words
            eval_token1 = unk_token if word1 not in known_words else word1
            eval_token2 = unk_token if word2 not in known_words else word2
            # log since probability becomes very small on multiplication
            log_review_prob += math.log(bigram_counts.get((eval_token1, eval_token2), 1)/token_counts[eval_token1])
        validation_probs_log.append(log_review_prob)
    perplexity = math.exp(-sum(validation_probs_log)/total_tokens)
    return perplexity


def evaluate_model(corpus: list, train_probs: dict, total_tokens_test: int):
    validation_probs_log = []
    for review in corpus:
        log_review_prob = 0
        for token in review:
            # log since probability becomes very small on multiplication
            log_review_prob += math.log(train_probs[token])
        validation_probs_log.append(log_review_prob)
    # TODO: Verify denominator -> train count or test count
    perplexity = math.exp(-sum(validation_probs_log)/total_tokens_test)
    return perplexity

def evaluate_bigram_model(corpus: list, train_probs: dict, total_tokens_test: int, prob_method, param):
    validation_probs_log = []
    for review in corpus:
        log_review_prob = 0
        for bigram in review:
            # if the key does not exist, we do not have probability of "word1 word2"
            # calculate it: prob_method(0, unigram_counts[bigram[0]], len(vocabulary), param)
            # log since probability becomes very small on multiplication
            log_review_prob += math.log(train_probs.get(bigram, prob_method(0, unigram_counts[bigram[0]], len(vocabulary), param)))
        validation_probs_log.append(log_review_prob)
    perplexity = math.exp(-sum(validation_probs_log)/total_tokens_test)
    return perplexity

'''
LOAD AND PREPROCESS TRAIN DATA
'''

train_unigrams_sentences = []
train_bigrams_sentences = []

for line in train_file:
    line = line.decode("utf-8")     # only for URL

    # preprocess
    line = preprocess_entry(line)
    line = line.split()

    # Add the start and end symbols
    line.insert(0, start_token)
    line.append(end_token)

    train_unigrams_sentences.append(line)

    previous_token = None    # bigram history
    current_bigrams = []

    for unigram in line:
        unigram_counts[unigram] = unigram_counts.get(unigram, 0) + 1

        if previous_token:
            bigram = (previous_token, unigram)
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            current_bigrams.append(bigram)
        previous_token = unigram
    train_bigrams_sentences.append(current_bigrams)

# initialise vocabulary
vocabulary = unigram_counts.keys()
total_tokens_train = sum(unigram_counts.values())

print('Vocab length:', len(vocabulary))

'''
LOAD AND PREPROCESS TEST DATA
'''
test_unigrams_sentences = []
test_bigrams_sentences = []

test_file = urllib.request.urlopen(test_url, context=context)
for line in test_file:
    line = line.decode("utf-8")     # only for URL

    # preprocess
    line = preprocess_entry(line)
    line = line.split()

    # replace unknown words with unknown token
    line = [unk_token if word not in vocabulary else word for word in line]

    # Add the start and end symbols
    line.insert(0, start_token)
    line.append(end_token)

    # add unigrams for this review
    test_unigrams_sentences.append(line)

    # add bigrams for this review
    previous_token = None    # bigram history
    current_bigrams = []
    for unigram in line:
        if previous_token:
            bigram = (previous_token, unigram)
            current_bigrams.append(bigram)
        previous_token = unigram
    test_bigrams_sentences.append(current_bigrams)

total_tokens_test = sum([len(review) for review in test_unigrams_sentences])

'''
Use <UNK> for count<=1, attempt to get perplexity
'''
unknown_tokens_train = set()
unigrams_counts_unk = copy.deepcopy(unigram_counts)
counter = 0
for unigram, count in unigram_counts.items():
    if count <=1:
        counter += count
        unknown_tokens_train.add(unigram)
        del unigrams_counts_unk[unigram]
# add our decided unknown token counts with <UNK>
unigrams_counts_unk[unk_token] = counter
known_words = set(vocabulary) - unknown_tokens_train

bigram_counts_unk = copy.deepcopy(bigram_counts)
for bigram, count in bigram_counts.items():
    if bigram[0] not in known_words and bigram[1] not in known_words:
        del bigram_counts_unk[bigram]
        bigram_counts_unk[(unk_token, unk_token)] = bigram_counts_unk.get((unk_token, unk_token), 0) + 1
    elif bigram[0] not in known_words:
        del bigram_counts_unk[bigram]
        bigram_counts_unk[(unk_token, bigram[1])] = bigram_counts_unk.get((unk_token, bigram[1]), 0) + 1
    elif bigram[1] not in known_words:
        del bigram_counts_unk[bigram]
        bigram_counts_unk[(bigram[0], unk_token)] = bigram_counts_unk.get((bigram[0], unk_token), 0) + 1


print('Total:', len(unigram_counts), 'UNK:', len(unknown_tokens_train), 'Known:', len(unigrams_counts_unk))


print('\n-----------------BASIC UNK PROCESSING----------------------')

print('Perplexity with UNK, unigram, train data:', evaluate_basic_model_unigram(train_unigrams_sentences, unigrams_counts_unk, total_tokens_train))
print('Perplexity with UNK, unigram, test data:', evaluate_basic_model_unigram(test_unigrams_sentences, unigrams_counts_unk, total_tokens_test))
print('Perplexity with UNK, bigram, train data:', evaluate_basic_model_bigram(train_bigrams_sentences, unigrams_counts_unk, bigram_counts_unk, total_tokens_train))
print('Perplexity with UNK, bigram, test data:', evaluate_basic_model_bigram(test_bigrams_sentences, unigrams_counts_unk, bigram_counts_unk, total_tokens_test))

'''
SMOOTHING PROCESSING
'''
# add unknown tokens FOR SMOOTHING
for unigram in vocabulary:
    bigram_counts[(unk_token, unigram)] = 0
    bigram_counts[(unigram, unk_token)] = 0
bigram_counts[(unk_token, unk_token)] = 0
unigram_counts[unk_token] = 0

# Reinit for smoothing
vocabulary = unigram_counts.keys()  # set of unique tokens
total_tokens_train = sum(unigram_counts.values())
vocabulary_size = len(vocabulary)

# Smoothing parameters
laplace_smoothing_param = 1
add_k_smoothing_param_1 = 0.1
add_k_smoothing_param_2 = 0.5

# Calculate unigram probabilities with Laplace Smoothing
unigram_probabilities_laplace = {word: laplace_smoothing(count, total_tokens_train, vocabulary_size, laplace_smoothing_param) for word, count in unigram_counts.items()}

# Calculate unigram probabilities with Add-k Smoothing (k = 0.1)
unigram_probabilities_add_k_1 = {word: add_k_smoothing(count, total_tokens_train, vocabulary_size, add_k_smoothing_param_1) for word, count in unigram_counts.items()}

# Calculate unigram probabilities with Add-k Smoothing (k = 0.5)
unigram_probabilities_add_k_2 = {word: add_k_smoothing(count, total_tokens_train, vocabulary_size, add_k_smoothing_param_2) for word, count in unigram_counts.items()}

# Calculate bigram probabilities with Laplace Smoothing
bigram_probabilities_laplace = {bigram: laplace_smoothing(count, unigram_counts[bigram[0]], vocabulary_size, laplace_smoothing_param) for bigram, count in bigram_counts.items()}

# Calculate bigram probabilities with Add-k Smoothing (k = 0.1)
bigram_probabilities_add_k_1 = {bigram: add_k_smoothing(count, unigram_counts[bigram[0]], vocabulary_size, add_k_smoothing_param_1) for bigram, count in bigram_counts.items()}

# Calculate bigram probabilities with Add-k Smoothing (k = 0.5)
bigram_probabilities_add_k_2 = {bigram: add_k_smoothing(count, unigram_counts[bigram[0]], vocabulary_size, add_k_smoothing_param_2) for bigram, count in bigram_counts.items()}


print('\n------------------TRAINING CORPUS--------------------------')
print('\nPerplexity of unigrams with Laplace smoothing:', evaluate_model(train_unigrams_sentences, unigram_probabilities_laplace, total_tokens_train))
print('\nPerplexity of unigram with Add-k smoothing k=0.1:', evaluate_model(train_unigrams_sentences, unigram_probabilities_add_k_1, total_tokens_train))
print('\nPerplexity of unigram with Add-k smoothing k=0.5:', evaluate_model(train_unigrams_sentences, unigram_probabilities_add_k_2, total_tokens_train))
print('\nPerplexity of bigrams with Laplace smoothing:', evaluate_bigram_model(train_bigrams_sentences, bigram_probabilities_laplace, total_tokens_train, laplace_smoothing, laplace_smoothing_param))
print('\nPerplexity of bigram with Add-k smoothing k=0.1:', evaluate_bigram_model(train_bigrams_sentences, bigram_probabilities_laplace, total_tokens_train, add_k_smoothing, add_k_smoothing_param_1))
print('\nPerplexity of bigram with Add-k smoothing k=0.5:',evaluate_bigram_model(train_bigrams_sentences, bigram_probabilities_laplace, total_tokens_train, add_k_smoothing, add_k_smoothing_param_2))

print('\n-------------------VALIDATION CORPUS-----------------------')
print('\nPerplexity of unigrams with Laplace smoothing:', evaluate_model(test_unigrams_sentences, unigram_probabilities_laplace, total_tokens_test))
print('\nPerplexity of unigram with Add-k smoothing k=0.1:', evaluate_model(test_unigrams_sentences, unigram_probabilities_add_k_1, total_tokens_test))
print('\nPerplexity of unigram with Add-k smoothing k=0.5:', evaluate_model(test_unigrams_sentences, unigram_probabilities_add_k_2, total_tokens_test))
print('\nPerplexity of bigrams with Laplace smoothing:', evaluate_bigram_model(test_bigrams_sentences, bigram_probabilities_laplace, total_tokens_test, laplace_smoothing, laplace_smoothing_param))
print('\nPerplexity of bigram with Add-k smoothing k=0.1:', evaluate_bigram_model(test_bigrams_sentences, bigram_probabilities_laplace, total_tokens_test, add_k_smoothing, add_k_smoothing_param_1))
print('\nPerplexity of bigram with Add-k smoothing k=0.5:',evaluate_bigram_model(test_bigrams_sentences, bigram_probabilities_laplace, total_tokens_test, add_k_smoothing, add_k_smoothing_param_2))

print('\n------------------------END--------------------------------')
