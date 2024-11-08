import os
import keras
from keras import layers, Model
import numpy as np
import io
import matplotlib.pyplot as plt
from keras.src.saving import load_model

# character used to represent the absence of input (first time step)
no_input_char = "\x00"
# maximum length of the input sequences; although RNNs allow for variable length sequences, we use a fixed length to speed up training
max_input_length = 40
# English text by Oscar Wilde
text_file_name = 'data/oscar-wilde.txt'
# path to the model file (if any)
model_file_name = 'data/wilde_language_model.keras'

def load_file(file_path: str) -> str:
    with io.open(file_path, encoding="utf-8") as f:
        file_contents = f.read().lower()
    return file_contents.replace("\n", " ")  # We remove newlines chars because they are not needed for this task


text = load_file(text_file_name)
print(f"Corpus length: {len(text):,} characters.")
vocabulary_chars = sorted(list(set(text + no_input_char)))
print(f"Unique chars (vocabulary size): {len(vocabulary_chars)}.")

# show the frequencies of all the chars in descending order
char_frequencies = {char: text.count(char) for char in vocabulary_chars}
sorted_char_frequencies = sorted(char_frequencies.items(), key=lambda item: item[1], reverse=True)
# visualize the frequencies of the characters as a histogram using matplotlib
plt.figure(figsize=(20, 5))
plt.bar([char for char, _ in sorted_char_frequencies], [freq for _, freq in sorted_char_frequencies])
plt.xlabel("Character")
plt.ylabel("Frequency")
plt.title("Character occurrences in the text")
plt.show()


char_to_index = dict((c, i) for i, c in enumerate(vocabulary_chars))
index_to_char = dict((i, c) for i, c in enumerate(vocabulary_chars))

def convert_X_and_y_to_one_hot(input_sequences_p: list, output_chars_p: list, max_input_length_p: int) \
        -> tuple[np.array, np.array]:
    """
    Convert the input and output sequences to one-hot encoding
    :param input_sequences_p: intput sequences in shape (num_sequences, max_input_length)
    :param output_chars_p: output characters in shape (num_sequences)
    :param max_input_length_p: the maximum length of the input sequences
    :return: (X_ds, y_ds) the input sequence in one-hot encoding and the output (next) characters in one-hot encoding
    with shapes (num_sequences, max_input_length, num_unique_chars) and (num_sequences, num_unique_chars) respectively
    """
    # We perform one-hot encoding of the input sequences and output characters
    # shape (num_sequences, max_input_length, num_unique_chars); all values are zeros
    # we use a fixed length of 40 characters for each sequence to speed up training
    X_ds = np.zeros((len(input_sequences_p), max_input_length_p, len(vocabulary_chars)), dtype="bool")
    # shape (num_sequences, num_unique_chars); all values are zeros
    y_ds = np.zeros((len(input_sequences_p), len(vocabulary_chars)), dtype="bool")
    # we set to 1 the corresponding character index for each sequence
    for sequence_index, input_sequence in enumerate(input_sequences_p):
        for char_index, sequence_char in enumerate(input_sequence):
            X_ds[sequence_index, char_index, char_to_index[sequence_char]] = 1
        y_ds[sequence_index, char_to_index[output_chars_p[sequence_index]]] = 1
    return X_ds, y_ds


def generate_X_and_y(text: str, max_input_length_p: int) -> tuple:
    """
    Generate the input and output sequences for the model
    :param text: the input text to be broken into sequences
    :param max_input_length_p: the maximum length of the input sequences
    :return: (X_ds, y_ds) the input sequence in one-hot encoding and the output (next) characters in one-hot encoding
    """
    input_sequences = []
    output_chars = []
    char_steps = 3
    # cut the text in semi-redundant sequences of maxlen characters
    for sequence_index in range(0, len(text) - max_input_length_p, char_steps):
        input_sequences.append(text[sequence_index: sequence_index + max_input_length_p])
        output_chars.append(text[sequence_index + max_input_length_p])
    print(f"Number of sequences: {len(input_sequences):,}.")
    # We perform one-hot encoding of the input sequences and output characters
    return convert_X_and_y_to_one_hot(input_sequences, output_chars, max_input_length_p)


X_dataset, y_dataset = generate_X_and_y(text, max_input_length)

language_model = keras.Sequential([
        keras.Input(shape=(max_input_length, len(vocabulary_chars))),
        layers.LSTM(128),
        layers.Dense(len(vocabulary_chars), activation="softmax"),
    ]
)
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
language_model.compile(loss="categorical_crossentropy", optimizer=optimizer)


def sample(predictions: np.array, temperature: float = 1.0) -> int:
    """
    Helper function to sample an index from a probability array
    :param predictions: the array of probabilities predicted by the model (softmax output)
    :param temperature: the temperature to apply to the probabilities (1=original, <1=conservative, >1=creative)
    :return: the index of the selected character
    """
    # convert to float64 to avoid numerical issues
    predictions = np.asarray(predictions).astype("float64")
    # modify the probabilities according to the temperature
    # calculate the log of the probabilities and divide by the temperature
    predictions = np.log(predictions) / temperature
    # apply the softmax function (exp(x) / sum(exp(x))) to the modified probabilities
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probas = np.random.multinomial(1, predictions, 1)  # probabilities after softmax (and temperature application)
    # return the index of the maximum value (the selected character)
    return np.argmax(probas)



# if we have a model, we load it from disk, otherwise we train it
if os.path.exists(model_file_name):
    language_model = load_model(model_file_name)
else:
    # we train the model and save it to disk
    epochs = 100
    batch_size = 128
    language_model.fit(X_dataset, y_dataset, batch_size=batch_size, epochs=epochs)
    language_model.save(model_file_name)


def prepare_input_text_to_one_hot(input_text_p: str) -> np.array:
    """
    Prepare the input text to be used as input for the language model
    :param input_text_p: the input text as str to be prepared
    :return: the one-hot encoded input text in shape (1, len(input_text), len(vocabulary_chars))
    """
    x_to_predict = np.zeros((1, len(input_text_p), len(vocabulary_chars)))
    for char_index, sequence_char in enumerate(input_text_p):
        x_to_predict[0, char_index, char_to_index[sequence_char]] = 1.0
    return x_to_predict


def generate_text(starting_text_p: str, model_p: Model, temperature_p: float = 1.0) -> str:
    """
    Generate text using the language model
    :param starting_text_p: the initial text to start the generation
    :param temperature_p: the temperature to apply to the predictions
    :return: the generated text
    """
    generated_text = ""
    input_text = starting_text_p
    for sequence_index in range(80):
        # We one-hot encode the input sequence
        x_to_predict = prepare_input_text_to_one_hot(input_text)
        probabilities_predictions = model_p.predict(x_to_predict, verbose=0)[0]
        next_index = sample(probabilities_predictions, temperature_p)
        next_char = index_to_char[next_index]
        # we update the input text with the next character predicted
        input_text = (input_text + next_char)[-max_input_length:]  # We keep only the last max_input_length chars
        generated_text += next_char
    return starting_text_p + generated_text


# We generate text for different starting texts and temperatures
starting_text_fragments = ["the ", no_input_char]
for starting_text in starting_text_fragments:
    print(f"Generating text for starting text: '{starting_text}'.")
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print("Temperature:", temperature)
        print("Generated text: ", generate_text(starting_text, language_model, temperature))
        print("-" * 80)



print("Probabilities of the first characters in the vocabulary:")
x_to_predict = prepare_input_text_to_one_hot(no_input_char)
probabilities_predictions = language_model.predict(x_to_predict, verbose=0)[0]
# short the characters by probability obtained in the prediction
sorted_characters = sorted(vocabulary_chars, key=lambda char: probabilities_predictions[char_to_index[char]], reverse=True)
for character in sorted_characters:
    print(f"Probability of '{character}': {probabilities_predictions[char_to_index[character]]:.8f}")


## Questions:
# 1. After all the information shown, do you have any idea that could be useful to improve the model?
# Answer: There are many characters thar are barely used in the text. That makes the vocubulary size bigger than it should be, implying more parameters.
# If we replace those characters with OOV, the model will be simpler, faster to train and probably more accurate.


def compute_probability_of_text(text_p: str, model_p: Model) -> float:
    """
    Compute the probability of a text using the language model
    :param text_p: the text for which we want to compute the probability
    :param model_p: the language model to use for the computation
    :return: the probability of the text for the given model
    """
    whole_input_text = no_input_char + text_p  # we pass the no_input_char to the model to get the prob of the first char
    probability = 1.0
    for sequence_index in range(len(text_p)):
        input_text, expected_char = whole_input_text[:sequence_index+1], whole_input_text[sequence_index+1]
        # We one-hot encode the input sequence
        x_to_predict = prepare_input_text_to_one_hot(input_text)
        probabilities_predictions = language_model.predict(x_to_predict, verbose=0)[0]
        probability *= probabilities_predictions[char_to_index[expected_char]]
    return probability

print("Probability of 'hello world' in the text: ", compute_probability_of_text("hello world", language_model), ".", sep="")
print("Probability of 'hola mundo' in the text: ", compute_probability_of_text("hola mundo", language_model), ".", sep="")
print("Probability of 'hi, dude' in the text: ", compute_probability_of_text("hi, dude", language_model), ".", sep="")


## Questions
# 2. Is 'hello world' more probable than 'hola mundo' in the text? Why?
# Answer: The probability of 'hello world' is higher than the probability of 'hola mundo' in the text because the text is in English and the model was trained on English text. The model has learned the patterns of the English language, so it assigns higher probabilities to English words and sequences of characters. Since 'hello world' is an English phrase, it is more likely to occur in the text than 'hola mundo', which is in Spanish. The model assigns lower probabilities to Spanish words and sequences of characters because it has not been trained on Spanish text. Therefore, the probability of 'hello world' is higher than the probability of 'hola mundo' in the text.
# 3. What is the probability of 'hi dude?' lower than the probability of 'hello world'?
# Answer: Because, although both are English phrases, 'what's up?' is less common than 'hello world' in the way Oscar Wilde writes.
# BTW, the longer the sentence, the lower the probability.
# 4. The probability of 'hello world' is very low. Does it mean that is not common / probable?
# Answer: No, it is low because it is computed the probability of that sentence in the whole English language (written by Oscar Wilde). That is why it is so low. However it is common in the English language (compare it to the two other sentences).
# BTW, the longer the sentence, the lower the probability.


