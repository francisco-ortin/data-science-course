# https://keras.io/examples/generative/lstm_character_level_text_generation/
# https://www.kaggle.com/code/stevengolo/character-level-language-modeling/code

import keras
from keras import layers
import numpy as np
import random
import io


# pip install datasets
from datasets import load_dataset  # from huggingface
dataset_name = "shibing624/source_code"

#train_dataset = load_dataset(dataset_name, 'python', split="train[:1000]")
#val_dataset = load_dataset(dataset_name, 'python', split="validation[:1000]")
#test_dataset = load_dataset(dataset_name, 'python', split="test[:1000]")


#train_text = "".join([row['text'] for row in train_dataset])
#val_text = "".join([row['text'] for row in val_dataset])
#test_text = "".join([row['text'] for row in test_dataset])

# make train_text to store the contents of util.py (open the file and read the contents)
"""with open('utils.py', 'r') as file:
    train_text = file.read()
print(train_text)
"""
"""path = keras.utils.get_file(
    "nietzsche.txt",
    origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
)
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display
"""

train_text = "import KKK\nimport keras\nimport numpy as np\nimport pandas as pd\n"*100


#print(f"Corpus length: train={len(train_text):,}, val={len(val_text):,}, test={len(test_text):,}.")
print(f"Corpus length: train={len(train_text):,}.")

vocabulary_chars = sorted(list(set(train_text)))
print("Total unique chars:", len(vocabulary_chars))
char_to_index = dict((char, index) for index, char in enumerate(vocabulary_chars))
index_to_char = dict((index, char) for index, char in enumerate(vocabulary_chars))

# cut the text in sequences of max_seq_length characters
lines = [line + "\n" for line in train_text.splitlines()]
input_sequences = []
output_chars = []
for line in lines:
    for char_index_in_line in range(1, len(line)):
        input_sequences.append(line[:char_index_in_line])
        output_chars.append(line[char_index_in_line])
print(f"Number of input sequences: {len(input_sequences):,}.")
print(f"Number of output chars: {len(output_chars):,}.")

# one-hot encode the sequences into binary arrays
# shape (num_sequences, max_seq_length, num_unique_chars); all values are zeros
X_train_ds = np.zeros((len(input_sequences), max_seq_length, len(vocabulary_chars)), dtype="bool")
# shape (num_sequences, num_unique_chars); all values are zeros
y_train_ds = np.zeros((len(input_sequences), len(vocabulary_chars)), dtype="bool")
# set the appropriate indices to 1 in each one-hot vector
for sequence_index, input_sequence in enumerate(input_sequences):
    for char_index, input_char in enumerate(input_sequence):
        # get the sequence, the char, and set the corresponding char index to 1
        X_train_ds[sequence_index, char_index, char_to_index[input_char]] = 1
    # get the output entry and set the corresponding char index to 1
    y_train_ds[sequence_index, char_to_index[output_chars[sequence_index]]] = 1

model = keras.Sequential([
        keras.Input(shape=(None, len(vocabulary_chars))),
        layers.LSTM(128),
        layers.Dense(len(vocabulary_chars), activation="softmax"),
    ]
)
optimizer = keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)


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


epochs = 3  # 40
batch_size = 128

model.fit(X_train_ds, y_train_ds, batch_size=batch_size, epochs=epochs)

start_index = random.randint(0, len(train_text) - max_seq_length - 1)
for temperature in [0.5, 1.0, 1.2]:
    print("Temperature:", temperature)

    generated = ""
    input_sequence = train_text[start_index: start_index + max_seq_length]
    input_sequence = train_text[:max_seq_length]
    input_sequence = "i"
    original_input_sequence = input_sequence
    print('Generating with seed: "' + input_sequence + '"')

    for sequence_index in range(20):
        x_pred = np.zeros((1, len(input_sequence), len(vocabulary_chars)))
        for char_index, input_char in enumerate(input_sequence):
            x_pred[0, char_index, char_to_index[input_char]] = 1.0
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = index_to_char[next_index]
        input_sequence = input_sequence[1:] + next_char
        generated += next_char

    print("Generated: ", original_input_sequence + generated)
