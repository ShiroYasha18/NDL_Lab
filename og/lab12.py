import numpy as np

# Part A: Single RNN Cell Forward Step
def rnn_cell_forward(x_t, h_prev, parameters):
    Wh = parameters['Wh']  # Hidden recurrent weights
    Wx = parameters['Wx']  # Input weights
    b = parameters['b']    # Bias
    
    # Forward step calculation
    h_next = np.tanh(np.dot(Wh, h_prev) + np.dot(Wx, x_t) + b)
    cache = (h_next, h_prev, x_t, parameters)
    
    return h_next, cache

# Part B: RNN Forward Propagation
def rnn_forward(x, h0, parameters):
    h = h0
    caches = []
    H = []  # Store all hidden states
    
    # Forward propagation through time
    for t in range(len(x)):
        h, cache = rnn_cell_forward(x[t], h, parameters)
        H.append(h)
        caches.append(cache)
    
    return np.array(H), caches

# Part C: LSTM Cell Implementation
def lstm_cell_forward(x_t, h_prev, c_prev, parameters):
    Wf = parameters['Wf']  # Forget gate weights
    Wi = parameters['Wi']  # Input gate weights
    Wc = parameters['Wc']  # Cell state weights
    Wo = parameters['Wo']  # Output gate weights
    bf = parameters['bf']  # Forget gate bias
    bi = parameters['bi']  # Input gate bias
    bc = parameters['bc']  # Cell state bias
    bo = parameters['bo']  # Output gate bias
    
    # Gates calculation
    ft = sigmoid(np.dot(Wf, np.concatenate([h_prev, x_t])) + bf)  # Forget gate
    it = sigmoid(np.dot(Wi, np.concatenate([h_prev, x_t])) + bi)  # Input gate
    cct = np.tanh(np.dot(Wc, np.concatenate([h_prev, x_t])) + bc)  # Candidate
    ot = sigmoid(np.dot(Wo, np.concatenate([h_prev, x_t])) + bo)  # Output gate
    
    # States calculation
    c_next = ft * c_prev + it * cct  # Next cell state
    h_next = ot * np.tanh(c_next)    # Next hidden state
    
    cache = (h_next, c_next, h_prev, c_prev, ft, it, cct, ot, parameters)
    
    return h_next, c_next, cache

# Helper function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Example usage
if __name__ == "__main__":
    # Initialize parameters
    hidden_size, input_size = 4, 3
    parameters = {
        'Wh': np.random.randn(hidden_size, hidden_size) * 0.01,
        'Wx': np.random.randn(hidden_size, input_size) * 0.01,
        'b': np.zeros((hidden_size, 1))
    }
    
    # Test RNN cell
    x_t = np.random.randn(input_size, 1)
    h_prev = np.zeros((hidden_size, 1))
    h_next, _ = rnn_cell_forward(x_t, h_prev, parameters)
    print("RNN Cell Output Shape:", h_next.shape)
    
    # Test RNN forward
    x = np.random.randn(5, input_size, 1)  # 5 time steps
    H, _ = rnn_forward(x, h_prev, parameters)
    print("RNN Forward Output Shape:", H.shape)
    
    # Test LSTM cell
    lstm_params = {
        'Wf': np.random.randn(hidden_size, hidden_size + input_size) * 0.01,
        'Wi': np.random.randn(hidden_size, hidden_size + input_size) * 0.01,
        'Wc': np.random.randn(hidden_size, hidden_size + input_size) * 0.01,
        'Wo': np.random.randn(hidden_size, hidden_size + input_size) * 0.01,
        'bf': np.zeros((hidden_size, 1)),
        'bi': np.zeros((hidden_size, 1)),
        'bc': np.zeros((hidden_size, 1)),
        'bo': np.zeros((hidden_size, 1))
    }
    c_prev = np.zeros((hidden_size, 1))
    h_next, c_next, _ = lstm_cell_forward(x_t, h_prev, c_prev, lstm_params)
    print("LSTM Cell Output Shape:", h_next.shape)