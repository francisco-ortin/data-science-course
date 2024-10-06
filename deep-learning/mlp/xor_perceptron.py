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
model.add(Dense(1, input_dim=2, activation='sigmoid'))
model.summary()  # params=3 (2 weights for inputs + bias)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
history = model.fit(X, y, epochs=100)

show_history(history, loss_label='Training loss', accuracy_label='Training accuracy')

prediction_probabilities = model.predict(X, verbose=0)
prediction = np.vectorize(lambda x: 1 if x > 0.5 else 0)(prediction_probabilities)

print('Accuracy:', accuracy_score(y, prediction))
print('F1-score:', f1_score(y, prediction))

print(f"Actual values: {y}.")
print(f"Predicted values: {prediction}.")


# Questions:
# 1) What is happening? Why
# Answer: The Perceptron NN is not able to solve the problem because it is non-linearly separable problem
#       and Perceptron can only solve linear problems (the same as Logistic Regression).