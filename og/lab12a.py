import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def softmax(x): e = np.exp(x - np.max(x)); return e / e.sum(axis=0)
def rnn_cell_forward(xt, a_prev, p):
    a_next = np.tanh(np.dot(p["Waa"], a_prev) + np.dot(p["Wax"], xt) + p["ba"])
    yt = softmax(np.dot(p["Wya"], a_next) + p["by"])
    return a_next, yt

# Dummy input
xt = np.random.randn(3, 1)
a_prev = np.zeros((5, 1))
p = {
    "Waa": np.random.randn(5, 5),
    "Wax": np.random.randn(5, 3),
    "ba": np.zeros((5, 1)),
    "Wya": np.random.randn(2, 5),
    "by": np.zeros((2, 1))
}

_, yt = rnn_cell_forward(xt, a_prev, p)
y_pred = np.argmax(yt)
y_true = 1  # Simulated label

print("\n🔹 RNN Cell (Q12.a) Accuracy Metrics:")
print(f"Accuracy:  {accuracy_score([y_true], [y_pred]):.2f}")
print(f"Precision: {precision_score([y_true], [y_pred]):.2f}")
print(f"Recall:    {recall_score([y_true], [y_pred]):.2f}")
print(f"F1 Score:  {f1_score([y_true], [y_pred]):.2f}")
