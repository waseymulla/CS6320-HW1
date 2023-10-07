import urllib.request
import re
import ssl
import math

# Define start and end symbols
start_symbol = "###"  # Symbol to mark the start of a data point
end_symbol = "$$$"    # Symbol to mark the end of a data point

# Function to generate bigrams from a list of words
def generate_bigrams(words):
    bigrams = []
    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        bigrams.append(bigram)
    return bigrams

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
# total_unigrams = 0
# vocabulary = set()
previous_word = None

'''
# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\']', '', text)
    return text
'''

def preprocess_entry(text: str):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)    # remove special characters
    text = re.sub(r'(\s+)', ' ', text)              # remove multiple whitespaces
    return text

# Laplace (Add-one) Smoothing Function
def laplace_smoothing(count, total, vocab_size, smoothing_param=1):
    return (count + smoothing_param) / (total + smoothing_param * vocab_size)

# Add-k Smoothing Function
def add_k_smoothing(count, unigram_count, vocab_size, smoothing_param):
    return (count + smoothing_param) / (unigram_count + smoothing_param * vocab_size)

def calculate_perplexity_corpus(probabililty_dict: dict):
    probabililty_list = probabililty_dict.values()
    return math.exp(-sum([math.log(value) for value in probabililty_list])/len(probabililty_list))

# Smoothing parameters
laplace_smoothing_param = 1
add_k_smoothing_param_1 = 0.1
add_k_smoothing_param_2 = 0.5

# Initialize a list to store unknown words
unknown_words = []

test_limit = 5

for line in train_file:
    line = line.decode("utf-8")     # only for URL

    if test_limit:
        print(test_limit, line)

    # preprocess
    line = preprocess_entry(line)
    line = line.split()

    # Add the start and end symbols
    line.insert(0, start_symbol)
    line.append(end_symbol)

    if test_limit:
        print(test_limit, line)
        test_limit -= 1

    previous_word = None    # bigram history
    for word in line:
        unigram_counts[word] = unigram_counts.get(word, 0) + 1

        if previous_word:
            bigram = (previous_word, word)
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        previous_word = word

# initialise vocabulary
vocabulary = unigram_counts.keys()
total_unigrams = len(vocabulary)

# Initialize a list to store unknown words in the test data
unknown_words_test = []

test_unigrams = {}
test_bigrams = {}

# Load and preprocess test data
test_file = urllib.request.urlopen(test_url, context=context)
for line in test_file:
    line = line.decode("utf-8")     # only for URL

    # preprocess
    line = preprocess_entry(line)
    line = line.split()
    
    # Add the start and end symbols
    line.insert(0, start_symbol)
    line.append(end_symbol)

    previous_word = None    # bigram history
    for word in line:
        test_unigrams[word] = test_unigrams.get(word, 0) + 1

        if previous_word:
            bigram = (previous_word, word)
            test_bigrams[bigram] = test_bigrams.get(bigram, 0) + 1
        previous_word = word

# Calculate unigram probabilities with Laplace Smoothing
unigram_probabilities_laplace = {word: laplace_smoothing(count, total_unigrams, len(vocabulary), laplace_smoothing_param) for word, count in unigram_counts.items()}

# Calculate unigram probabilities with Add-k Smoothing (k = 0.1)
unigram_probabilities_add_k_1 = {word: add_k_smoothing(count, total_unigrams, len(vocabulary), add_k_smoothing_param_1) for word, count in unigram_counts.items()}

# Calculate unigram probabilities with Add-k Smoothing (k = 0.5)
unigram_probabilities_add_k_2 = {word: add_k_smoothing(count, total_unigrams, len(vocabulary), add_k_smoothing_param_2) for word, count in unigram_counts.items()}

# Calculate bigram probabilities with Laplace Smoothing
bigram_probabilities_laplace = {bigram: laplace_smoothing(count, unigram_counts[bigram[0]], len(vocabulary), laplace_smoothing_param) for bigram, count in bigram_counts.items()}

# Calculate bigram probabilities with Add-k Smoothing (k = 0.1)
bigram_probabilities_add_k_1 = {bigram: add_k_smoothing(count, unigram_counts[bigram[0]], len(vocabulary), add_k_smoothing_param_1) for bigram, count in bigram_counts.items()}

# Calculate bigram probabilities with Add-k Smoothing (k = 0.5)
bigram_probabilities_add_k_2 = {bigram: add_k_smoothing(count, unigram_counts[bigram[0]], len(vocabulary), add_k_smoothing_param_2) for bigram, count in bigram_counts.items()}

# Print the results
print('Total unigrams (Train):', total_unigrams)

print('\nUnigram probabilities (Laplace Smoothing):')
print(unigram_probabilities_laplace)

print('\nUnigram probabilities (Add-k Smoothing with k = 0.1):')
print(unigram_probabilities_add_k_1)

print('\nUnigram probabilities (Add-k Smoothing with k = 0.5):')
print(unigram_probabilities_add_k_2)

print('\nBigram probabilities (Laplace Smoothing):')
print(bigram_probabilities_laplace)

print('\nBigram probabilities (Add-k Smoothing with k = 0.1):')
print(bigram_probabilities_add_k_1)

print('\nBigram probabilities (Add-k Smoothing with k = 0.5):')
print(bigram_probabilities_add_k_2)

# print('\nUnknown words (Train):')
# print(unknown_words)

print('\nUnknown words (Test):')
print(unknown_words_test)

print('\nPerplexity unigram #1:', calculate_perplexity_corpus(unigram_probabilities_laplace))
print('\nPerplexity unigram #2:', calculate_perplexity_corpus(unigram_probabilities_add_k_1))
print('\nPerplexity unigram #3:', calculate_perplexity_corpus(unigram_probabilities_add_k_2))
print('\nPerplexity bigram #1:', calculate_perplexity_corpus(bigram_probabilities_laplace))
print('\nPerplexity bigram #2:', calculate_perplexity_corpus(bigram_probabilities_add_k_1))
print('\nPerplexity bigram #3:', calculate_perplexity_corpus(bigram_probabilities_add_k_2))

print('----END----')
# print(vocabulary)
