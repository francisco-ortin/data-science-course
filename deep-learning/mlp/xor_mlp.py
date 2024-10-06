from keras.src.layers import Dense
from keras.src.optimizers import SGD, Adam
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
import numpy as np

from anns.utils import show_history

# Two independent variables of the XOR problem
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
# Dependent variable (XOR value)
y = np.array([[0], [1], [1], [0]])

model = Sequential()
# One hidden layer with 4 neurons (4 x (2+1) = 12 params)
model.add(Dense(4, input_dim=2, activation='sigmoid'))
# Output layer (1 neuron) (1 x (4+1) = 5 params)
model.add(Dense(1, activation='sigmoid'))  # We do not need to include the input_dim (is inferred)

model.summary()  # params=17

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
history = model.fit(X, y, epochs=200)

show_history(history, loss_label='Training loss', accuracy_label='Training accuracy')

prediction_probabilities = model.predict(X, verbose=0)
prediction = np.vectorize(lambda x: 1 if x > 0.5 else 0)(prediction_probabilities)

print('Accuracy:', accuracy_score(y, prediction))
print('F1-score:', f1_score(y, prediction))

print(f"Actual values: {y}.")
print(f"Predicted values: {prediction}.")


# Questions:
# 1) What happens if you decrease the number of neurons of the hidden layer to 2? Why?
#    Answer: it gets the same accuracy, but it is trained faster. It is because it has fewer parameters
#    to lean and the network is powerful enough to solve the problem.
# 2) What happens if you decrease the number of neurons of the hidden layer to 1? Why?
#    Answer: the network is not sufficiently powerful to solve the problem => underfitting
