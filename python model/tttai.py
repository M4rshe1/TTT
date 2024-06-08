from sklearn.model_selection import train_test_split
from keras.layers import Input
import keras
import numpy as np
import json
import os


def create_model():
    input_layer = Input(shape=(27,))
    dense1 = keras.layers.Dense(45, activation='relu')(input_layer)
    dense2 = keras.layers.Dense(65, activation='relu')(dense1)
    dense3 = keras.layers.Dense(45, activation='relu')(dense2)
    dense4 = keras.layers.Dense(16, activation='relu')(dense3)
    output_layer = keras.layers.Dense(9, activation='linear')(dense4)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def reverse_convert_board(board):
    new_board = []
    for i in range(0, 27, 3):
        if board[i] == 1:
            new_board.append(0)
        elif board[i + 1] == 1:
            new_board.append(1)
        else:
            new_board.append(-1)
    return np.array(new_board)


# make a function to convert the ai output to the same format as expected output
def convert_predictions(predictions, threshold=0.5):
    # converted_predictions = np.where(np.abs(predictions) < threshold, 0, np.round(predictions))
    # clipped_predictions = np.clip(converted_predictions, -1, 1)
    # return clipped_predictions.astype(int)

    # get the one with the highest value
    return np.where(predictions == np.max(predictions), 1, 0)


def train_model(model, states, next_states, samples):
    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(states, next_states, test_size=0.2, random_state=42)

    # Fit the model
    model.fit(x_train, y_train, epochs=32, batch_size=32, validation_data=(x_test, y_test))

    # Evaluate the model
    loss = model.evaluate(x_test, y_test)

    print(f'Test Loss: {loss}')
    # what it the accuracy of the model?
    print(f'Test Accuracy: {1 - loss}')

    test_model(model, states, next_states)

    # Print the predictions, input and expected output
    if not os.path.exists(f"models/{samples}"):
        os.makedirs(f"models/{samples}")

    model.save(f'models/{samples}/model.h5')


def test_model(model, states, next_states):
    valid = 0
    cinda_valid = 0
    samples = 5000
    _, x_test, _, y_test = train_test_split(states, next_states, test_size=0.2, random_state=42)

    predictions = model.predict(x_test)
    samples = min(samples, len(predictions))
    for i in range(samples):
        input_board = reverse_convert_board(x_test[i])
        expected_output = y_test[i]
        conv_pred = convert_predictions(predictions[i])
        same = True
        for j in range(9):
            if expected_output[j] != conv_pred[j]:
                same = False
                break
        cinda = np.sum(expected_output) == np.sum(conv_pred)
        not_valid = False

        for j in range(9):
            if input_board[j] != 0 and expected_output[j] == 1:
                not_valid = True
                break
        if cinda and not not_valid:
            cinda = True
            cinda_valid += 1
        else:
            cinda = False
        if same:
            valid += 1

        if same:
            continue
        print(f'index           : {i}')
        print(f'prediction      : {predictions[i]}')
        print(f'Input           : {input_board}')
        print(f'Expected Output : [ {expected_output[0]}  {expected_output[1]}  {expected_output[2]}  {expected_output[3]}  {expected_output[4]}  {expected_output[5]}  {expected_output[6]}  {expected_output[7]}  {expected_output[8]}]')
        print(f'conv pred       : [ {conv_pred[0]}  {conv_pred[1]}  {conv_pred[2]}  {conv_pred[3]}  {conv_pred[4]}  {conv_pred[5]}  {conv_pred[6]}  {conv_pred[7]}  {conv_pred[8]}]')
        print(f'valid           : {not not_valid}')
        print(f'match           : {same}')
        print(f'cinda valid     : {cinda}')
        print('-----------------')

    print(f'Cinda Valid predictions: {cinda_valid}/{samples}')
    print(f'Accuracy cinda: {cinda_valid / samples}')
    print(f'Valid predictions: {valid}/{samples}')
    print(f'Accuracy: {valid / samples}')


def main(samples):
    if os.path.exists(f'models/{samples}/model.h5'):
        print(f'Model with {samples} samples already exists')
        # load the model and test it
        model = keras.models.load_model(f'models/{samples}/model.h5')
        with open(f'datasets/{samples}_samples.json', 'r') as f:
            dataset = json.load(f)
        states = np.array(dataset['states'])
        next_states = np.array(dataset['next_states'])
        test_model(model, states, next_states)
        return
    # Load the dataset
    with open(f'datasets/{samples}_samples.json', 'r') as f:
        dataset = json.load(f)

    states = np.array(dataset['states'])
    next_states = np.array(dataset['next_states'])

    # Create the model
    model = create_model()

    # Train the model
    train_model(model, states, next_states, samples)

    # Save the model
