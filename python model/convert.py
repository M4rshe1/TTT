import tensorflow as tf
import numpy as np
import os


def export_weights(samples):
    if os.path.exists(f'models/{samples}/weights'):
        return

    # Specify the path to your HDF5 file containing the model
    current_path = os.path.dirname(os.path.abspath(__file__))
    model_file_path = os.path.join(current_path, f'models/{samples}/model.h5')

    # Load the model from the .h5 file
    model = tf.keras.models.load_model(model_file_path)

    # Extract weights and biases
    weights = []
    biases = []

    for layer in model.layers:
        if hasattr(layer, 'weights') and layer.weights:
            weight, bias = layer.get_weights()
            weights.append(weight)
            biases.append(bias)

    if not os.path.exists(f'models/{samples}/weights'):
        os.makedirs(f'models/{samples}/weights')
    # Save weights and biases to separate files or handle as desired
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        # Assuming you want to save them as CSV files
        weight_file_name = f'models/{samples}/weights/weight_layer_{i}.csv'
        bias_file_name = f'models/{samples}/weights/bias_layer_{i}.csv'
        np.savetxt(weight_file_name, weight, delimiter=';')
        np.savetxt(bias_file_name, bias, delimiter=';')
