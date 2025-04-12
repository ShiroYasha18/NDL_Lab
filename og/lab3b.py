from sklearn.neural_network import MLPClassifier

X, y = [[0,0], [0,1], [1,0], [1,1]], [0, 1, 1, 0]  # XOR data

# Test different learning rates
for lr in [0.001, 0.01, 0.1, 0.5]:
    model = MLPClassifier(
        hidden_layer_sizes=(2,), 
        activation='logistic',  # Sigmoid
        learning_rate_init=lr,
        max_iter=10000,
        random_state=1
    ).fit(X, y)
    print(f"LR: {lr} → Accuracy: {100 * model.score(X, y):.0f}%")