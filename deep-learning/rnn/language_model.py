import keras
from keras import layers
import numpy as np
import random
import io


# pip install datasets
from datasets import load_dataset  # from huggingface
dataset_name = "shibing624/source_code"

train_dataset = load_dataset(dataset_name, 'python', split="train[:1000]")
val_dataset = load_dataset(dataset_name, 'python', split="validation[:1000]")
test_dataset = load_dataset(dataset_name, 'python', split="test[:1000]")


train_text = "".join([row['text'] for row in train_dataset])
val_text = "".join([row['text'] for row in val_dataset])
test_text = "".join([row['text'] for row in test_dataset])

"""path = keras.utils.get_file(
    "nietzsche.txt",
    origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt",
)
with io.open(path, encoding="utf-8") as f:
    text = f.read().lower()
text = text.replace("\n", " ")  # We remove newlines chars for nicer display
"""


print(f"Corpus length: train={len(train_text):,}, val={len(val_text):,}, test={len(test_text):,}.")

unique_chars = sorted(list(set(train_text)))
print("Total unique chars:", len(unique_chars))
char_to_index = dict((char, index) for index, char in enumerate(unique_chars))
index_to_char = dict((index, char) for index, char in enumerate(unique_chars))

# cut the text in sequences of maxlen characters
max_seq_length = 40
input_sequences = []
output_chars = []
for sequence_index in range(0, len(train_text) - max_seq_length):
    input_sequences.append(train_text[sequence_index: sequence_index + max_seq_length])
    output_chars.append(train_text[sequence_index + max_seq_length])
print("Number of input sequences:", len(input_sequences))
print("Number of output chars:", len(output_chars))

# one-hot encode the sequences into binary arrays
# shape (num_sequences, max_seq_length, num_unique_chars); all values are zeros
X_train_ds = np.zeros((len(input_sequences), max_seq_length, len(unique_chars)), dtype="bool")
# shape (num_sequences, num_unique_chars); all values are zeros
y_train_ds = np.zeros((len(input_sequences), len(unique_chars)), dtype="bool")
# set the appropriate indices to 1 in each one-hot vector
for sequence_index, input_sequence in enumerate(input_sequences):
    for char_index, input_char in enumerate(input_sequence):
        # get the sequence, the char, and set the corresponding char index to 1
        X_train_ds[sequence_index, char_index, char_to_index[input_char]] = 1
    # get the output entry and set the corresponding char index to 1
    y_train_ds[sequence_index, char_to_index[output_chars[sequence_index]]] = 1

model = keras.Sequential([
        keras.Input(shape=(max_seq_length, len(unique_chars))),
        layers.LSTM(128),
        layers.Dense(len(unique_chars), activation="softmax"),
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


epochs = 2  # 40
batch_size = 128

for epoch in range(epochs):
    model.fit(X_train_ds, y_train_ds, batch_size=batch_size, epochs=1)
    print()
    print("Generating text after epoch: %d" % epoch)

    start_index = random.randint(0, len(train_text) - max_seq_length - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("...Diversity:", diversity)

        generated = ""
        input_sequence = train_text[start_index: start_index + max_seq_length]
        print('...Generating with seed: "' + input_sequence + '"')

        for sequence_index in range(400):
            x_pred = np.zeros((1, max_seq_length, len(unique_chars)))
            for char_index, input_char in enumerate(input_sequence):
                x_pred[0, char_index, char_to_index[input_char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = index_to_char[next_index]
            input_sequence = input_sequence[1:] + next_char
            generated += next_char

        print("...Generated: ", generated)
        print("-")