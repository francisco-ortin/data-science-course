import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

# Example text corpus (you can replace this with your own text)
text = "import keras\nimport numpy as np\nimport pandas as pd\nimport utils\nimport matplotlib as mt\n" * 1

# Create a set of characters (vocabulary)
chars = sorted(set(text))
char_to_index = {char: idx for idx, char in enumerate(chars)}
index_to_char = {idx: char for idx, char in enumerate(chars)}

# Max sequence length (you can adjust this)
max_length = 5  # Can be adjusted to whatever max length you need

# Prepare training data: sequences of length 0 to max_length
X_train = []
y_train = []

# Input length varies from 0 to max_length
for i in range(len(text) - 2):
    for seq_len in range(1, max_length + 1):  # Create sequences of various lengths
        if i + seq_len + 1 < len(text):  # Ensure we don't go out of bounds
            # Create input sequence of length seq_len
            input_seq = text[i:i + seq_len]
            target_char = text[i + seq_len]  # Target is the next character

            # Convert input sequence to indices
            input_indices = [char_to_index[char] for char in input_seq]
            target_index = char_to_index[target_char]

            # Append to training data
            X_train.append(input_indices)
            y_train.append(target_index)

# Pad sequences to ensure they all have the same length (max_length)
X_train = pad_sequences(X_train, maxlen=max_length, padding='pre')

# One-hot encode the labels for the characters
y_train = keras.utils.to_categorical(y_train, num_classes=len(chars))

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(chars), output_dim=50, input_length=max_length))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))  # Output layer with softmax

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10)


# Function to generate next character probabilities with character labels
def predict_next_chars(input_str):
    # Convert input string to indices
    input_indices = [char_to_index[char] for char in input_str]
    input_indices = np.array(input_indices).reshape(1, -1)

    # Pad sequences to length `max_length` (for handling empty input and shorter inputs)
    input_indices = pad_sequences(input_indices, maxlen=max_length, padding='pre')

    # Get the model prediction (probabilities for all characters)
    predictions = model.predict(input_indices, verbose=0)[0]

    # Return the probabilities with associated characters
    prob_char_list = [(index_to_char[idx], prob) for idx, prob in enumerate(predictions)]

    # Sort by probability in descending order for better clarity
    prob_char_list = sorted(prob_char_list, key=lambda x: x[1], reverse=True)

    return prob_char_list


# Example predictions:
print("Probabilities for the first character:")
for char, prob in predict_next_chars("")[:10]:  # Top 10 predictions
    print(f"Char: {char}, Probability: {prob:.4f}")

print("\nProbabilities for the second character after 'h':")
for char, prob in predict_next_chars("h")[:10]:  # Top 10 predictions
    print(f"Char: {char}, Probability: {prob:.4f}")

print("\nProbabilities for the next character after 'he':")
for char, prob in predict_next_chars("he")[:10]:  # Top 10 predictions
    print(f"Char: {char}, Probability: {prob:.4f}")
