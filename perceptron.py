threshold = 0.5
gamma = 0.1
weights = [0, 0, 0]
training_dataset = [((1, 0, 0), 1), ((1, 0, 1), 1), ((1, 1, 0), 1), ((1, 1, 1), 0)]

def scalar_product(inputs, weights):
    return sum(input_value * weight for input_value, weight in zip(inputs, weights))

while True:
    print('-' * 60)
    error_counter = 0
    for input_vector, expected_output in training_dataset:
        print('these are the weights %s' % weights)
        result = scalar_product(input_vector, weights) > threshold
        error = expected_output - result
        if error != 0:
            error_counter += 1
            for index, value in enumerate(input_vector):
                weights[index] += gamma * error * value
    if contador_de_errores == 0:
        break
