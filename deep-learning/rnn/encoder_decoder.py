import numpy as np
from keras import Model
import tensorflow as tf
import os
from tensorflow.keras.models import load_model


## PARAMETERS

vocab_size = 1_000
max_length = 50
chars_to_remove = ["¡", "¿"]
train_size_percentage = 85
embedding_size = 128
n_epochs = 10
n_lstm_units = 512
SOS_word, EOS_word = "startofsentence", "endofsentence"
model_file_name = 'data/english_spanish_encoder_decoder.keras'


## PREPARE THE DATASET

# read the contents of the data/english-spanish.txt file
with open("data/english-spanish.txt", 'r', encoding='utf-8') as file:
    text = file.read()
# remove the special characters
for special_char in chars_to_remove:
    text = text.replace(special_char, "")
# take the English and Spanish sentences, by splitting each line by the tab character
pairs: list[(str, str)] = [line.split("\t") for line in text.splitlines()]
np.random.shuffle(pairs)
# tales a list of pairs and returns a pair of lists: one with the English sentences and one with the Spanish sentences
sentences_en, sentences_es = zip(*pairs)

assert (n_sentences := len(sentences_en)) == len(sentences_es)
print(f"Number of sentences: {n_sentences:,}.")

print("Some example translations:")
for i in range(5):
    print(f"\t{i+1}: {sentences_en[i]} -> {sentences_es[i]}")


# TextVectorization is as keras layer that converts a batch of strings into either a list of token indices (ints)
# It could also output a dense representation of the strings, where each token is represented by a dense vector
text_vec_layer_en = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)
text_vec_layer_es = tf.keras.layers.TextVectorization(vocab_size, output_sequence_length=max_length)
# adapt makes the layer to transform each input sentence into a list of word indices, considering the vocabulary size
text_vec_layer_en.adapt(sentences_en)
# we adapt the Spanish  layer to the Spanish sentences, including the start and end of sentence tokens
text_vec_layer_es.adapt([f"{SOS_word} {sentence} {EOS_word}" for sentence in sentences_es])


print(f"Some example English words: {text_vec_layer_en.get_vocabulary()[:10]}")  # 0 is padding, visualized as ''
print(f"Some example Spanish words: {text_vec_layer_es.get_vocabulary()[:10]}")

# we split the data into training and validation sets
# we first take the input for the Encoder (English sentences)
X_train_encoder = tf.constant(sentences_en[:n_sentences * train_size_percentage // 100])
X_valid_encoder = tf.constant(sentences_en[n_sentences*train_size_percentage//100:])
# then, we take the input for the Decoder (Spanish sentences)
# We include the SOS at the beginning of each sentence. This is because we want the Decoder to start generating
# the first Spanish word, by passing SOS as the first input. Then, the Decoder will generate the first word and
# we will pass it to the Decoder again, so it can generate the second word, and so on, until it generates the EOS.
# EOS does not need to be added to the input, since we want the Decoder to generate it (it will be added to
# Y training dataset).
X_train_decoder = tf.constant([f"{SOS_word} {sentence}" for sentence in
                               sentences_es[:n_sentences * train_size_percentage // 100]])
X_valid_decoder = tf.constant([f"{SOS_word} {sentence}" for sentence in
                               sentences_es[n_sentences * train_size_percentage // 100:]])
# The output of the Decoder is the same as the input, but shifted one position to the right, and with EOS at the end
# of each sentence. This is because we want the Decoder to generate the first word of the Spanish sentence, then
# the second word, and so on, until it generates the EOS.
Y_train = text_vec_layer_es([f"{sentence} {EOS_word}" for sentence in sentences_es[:n_sentences * train_size_percentage // 100]])
Y_valid = text_vec_layer_es([f"{sentence} {EOS_word}" for sentence in sentences_es[n_sentences * train_size_percentage // 100:]])


## BUILD THE MODEL

def create_model(n_lstm_units_p: int, vocab_size_p: int) -> Model:
    """
    Creates a Keras model for the Encoder-Decoder architecture.
    :param n_lstm_units_p: Number of LSTM units in the Encoder and Decoder.
    :param vocab_size_p: Vocabulary size.
    :return: The model
    """
    # Both the Encoder and the Decoder will receive a batch of sentences as input (English sentences for the Encoder,
    # and Spanish sentences for the Decoder).
    encoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
    decoder_inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)

    # The input sequences are converted into lists of word indices using the TextVectorization layers
    encoder_input_ids = text_vec_layer_en(encoder_inputs)
    decoder_input_ids = text_vec_layer_es(decoder_inputs)

    # The word indices are then converted into dense vectors using an Embedding layer of `embedding_size` dimensions
    # The padding character zero is masked out, so it is ignored by the model (its weight is not updated/learned)
    encoder_embedding_layer = tf.keras.layers.Embedding(vocab_size_p, embedding_size, mask_zero=True)
    decoder_embedding_layer = tf.keras.layers.Embedding(vocab_size_p, embedding_size, mask_zero=True)
    # we connect the embedding layers to the input indices
    encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
    decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

    # We create the Encoder as a single bidirectional LSTM layer with half of the units (two LSTMs, one for each direction)
    # Return_state=True => gets a reference to the layer’s final state
    # Since we are using a bi-LSTM layer, the final state is a tuple containing 2 short- and 2 long-term states,
    # one pair for each direction (that is why we use *encoder_states)
    encoder = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(n_lstm_units_p // 2, return_state=True))
    encoder_outputs, *encoder_states = encoder(encoder_embeddings)

    # we concatenate the states of the left and right LSTMs (first, the 2 short-term states and then the 2 long-term states)
    encoder_state = [tf.concat([encoder_states[0], encoder_states[2]], axis=-1),  # short-term (0 & 2)
                     tf.concat([encoder_states[1], encoder_states[3]], axis=-1)]  # long-term (1 & 3)

    # The Decoder is also an LSTM layer with `n_lstm_units` units, but it returns sequences (return_sequences=True)
    # instead of the final state. It cannot be bidirectional, since it needs to generate the words in order (it would be cheating).
    # Remember that the Decoder is a conditional language model, so it needs to receive the states of the Encoder
    # (initial_state parameter)
    decoder = tf.keras.layers.LSTM(n_lstm_units_p, return_sequences=True)
    decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

    # We add a Dense layer with a softmax activation function to predict the next word in the Spanish sentence
    output_layer = tf.keras.layers.Dense(vocab_size_p, activation="softmax")
    Y_proba = output_layer(decoder_outputs)

    # Finally, we create the Keras Model, specifying the inputs and outputs
    model_loc = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[Y_proba])
    model_loc.summary()
    return model_loc


def compile_and_train_model(model: Model, X_train_encoder_p: np.array, X_train_decoder_p: np.array,
                            Y_train_p: np.array, X_valid_encoder_p: np.array, X_valid_decoder_p: np.array,
                            Y_valid_p: np.array, n_epochs_p: int, model_file_name: str) -> Model:
    if os.path.exists(model_file_name):
        return load_model(model_file_name)
    # we compile and train the model with sparse_categorical_crossentropy as the loss function, since the targets are integers
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    model.fit((X_train_encoder_p, X_train_decoder_p), Y_train_p,
          epochs=n_epochs_p, batch_size=32,
          validation_data=((X_valid_encoder_p, X_valid_decoder_p), Y_valid_p),
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])
    model.save(model_file_name)
    return model


model = create_model(n_lstm_units, vocab_size)
model = compile_and_train_model(model, X_train_encoder, X_train_decoder, Y_train, X_valid_encoder,
                                X_valid_decoder, Y_valid, n_epochs, model_file_name)

## INFERENCE

def translate(sentence_en: str) -> str:
    """
    Translates an English sentence into Spanish, preparing the input for the model and calling the predict method.
    :param sentence_en: The English sentence to translate.
    :return: The Spanish translation.
    """
    translation = ""
    for word_idx in range(max_length):
        # Encoder input: one English sentence (batch size = 1)
        X_inf_encoder = np.array([sentence_en])
        # Decoder input: SOS + existing translation (empty at the beginning)
        X_inf_decoder = np.array([SOS_word + translation])
        # We call predict with (Encoder_input, Decoder_input) to get the probabilities of the next word
        # we take the first sentence ([0]) and the probabilities idx-th word (returns a list of probabilities for max_length words)
        y_proba = model.predict((X_inf_encoder, X_inf_decoder), verbose=0)[0, word_idx]  # probas of the last predicted word
        # we take the word id with the highest probability
        predicted_word_id = np.argmax(y_proba)
        # we get the word from the vocabulary
        predicted_word = text_vec_layer_es.get_vocabulary()[predicted_word_id]
        if predicted_word == EOS_word:
            # we are done when we predict the end of sentence token
            break
        translation += " " + predicted_word
    return translation.strip()


# we test the translation with some sentences. Feel free to add more sentences to test the model
english_sentences = ["hello everyone",
                     "how old are you?",
                     "what is your name?",
                     "where are you from?",
                     "I like soccer",
                     "This is a too long sentence to be translated correctly"]
for sentence in english_sentences:
    print(f"{sentence} -> {translate(sentence)}.")


## BEAM SEARCH


def beam_search(sentence_en: str, beam_width: int, verbose=False):
    # Translation of the first word
    # Encoder input: one English sentence (batch size = 1)
    X_inf_encoder = np.array([sentence_en])
    # Decoder input: SOS
    X_inf_decoder = np.array([SOS_word])
    # Predict the probabilities of the first word
    y_proba = model.predict((X_inf_encoder, X_inf_decoder), verbose=0)[0, 0]  # first token's probas
    # we take the top k words with the highest probabilities Dict{word_id: proba}
    top_k_words = tf.math.top_k(y_proba, k=beam_width)
    # list of best (log_proba, translation) pairs
    # Important: instead of taking Prob(w1) * Prob(w2) * ... * Prob(wn), we take the log of the product:
    # log(Prob(w1)) + log(Prob(w2)) + ... + log(Prob(wn))
    # this is because the product of many probabilities between 0 and 1 can be very small and lead to 0.0 after some iterations
    top_translations = [
        (np.log(word_proba), text_vec_layer_es.get_vocabulary()[word_id])
        for word_proba, word_id in zip(top_k_words.values, top_k_words.indices)
    ]

    # displays the top first words if verbose mode
    print("Top first words:", top_translations) if verbose else None

    # Translation of the next words (from 1 on)
    for idx in range(1, max_length):
        # list of best (log_proba, translation) pairs
        candidates: list[(float, str)] = []
        for log_proba, translation in top_translations:
            if translation.endswith(EOS_word):
                candidates.append((log_proba, translation))
                # translation is finished, so don't try to extend it
                continue
            # Encoder input: one English sentence (batch size = 1)
            X_inf_encoder = np.array([sentence_en])  # encoder input
            # Decoder input: SOS + existing translation
            X_inf_decoder = np.array([SOS_word + " " + translation])  # decoder input
            # probabilites of the new word
            y_proba = model.predict((X_inf_encoder, X_inf_decoder), verbose=0)[0, idx]  # last token's proba
            # we include in candidates the top k existing translations with all the possible next words and their probabilities
            for word_id, word_proba in enumerate(y_proba):
                word = text_vec_layer_es.get_vocabulary()[word_id]
                candidates.append((log_proba + np.log(word_proba), f"{translation} {word}"))
        # we sort the candidates by the log of the probabilities and take the top k
        top_translations = sorted(candidates, reverse=True)[:beam_width]

        # displays the top translation so far, if verbose mode
        print("Top translations so far:", top_translations) if verbose else None

        # the process terminates when all the K top translations end with the EOS token
        if all([top_translation.endswith(EOS_word) for _, top_translation in top_translations]):
            # returns the best translation pair ([0] because it is sorted by log probabilities),
            # take the translation text ([1]) and remove the EOS token
            return top_translations[0][1].replace(EOS_word, "").strip()


for sentence in english_sentences:
    print("-" * 50)
    print(f"Translation with beam search for: \n\t '{sentence}':")
    translation = beam_search(sentence, 3, verbose=True)
    print(f"Spanish translation: {translation}.")

# This simple model performs decently on short sentences, but it struggles with longer sentences.
# It is possible to significantly improve the translation quality by using attention.
# A more sophisticated implementation of the Encoder-Decoder architecture with attention called Transformer
# is the state-of-the-art model for machine translation, currently used by GPT, BERT, and many other models.

## QUESTIONS
# 1. What would happen in beam search if we used probability product instead of the sum of log probabilities?
#    The product of many probabilities between 0 and 1 can be very small and lead to 0.0 after some iterations.
# 2. Why do you think the model is not able to translate `Hello` correctly?
#    The vocabulary is limited to 1,000 words, so it is possible that the word `Hello` is not in the vocabulary
#    because it was not very common in the training data.
# 3. Try it out.
#    We increase the vocabulary size from 1,000 to 5,000 and see if the translation improves.
# 4. Do you think the last sentence will be translated better with k=10? Try it out.
#    It might be because it is long and there might be a better global solution.
#    However, we tried it out and the translation is not better with k=10.


