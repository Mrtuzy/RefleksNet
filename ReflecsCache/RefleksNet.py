import pickle
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class ReflexNet:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500)
        self.trained = False

    def fit(self, X, y):
        # .pkl dosyasından verileri yükle
        with open("train_data.pkl", "rb") as f:
            data = pickle.load(f)
        X = data["X"]  # Özellik vektörleri
        y = data["y"]  # Aksiyon etiketleri

        # Modeli eğit (örnek: sklearn DecisionTreeClassifier)
        self.model = DecisionTreeClassifier()
        self.model.fit(X, y)

    def predict(self, state):
        # state dict'ini uygun feature array'e çevir
        features = np.array([
            state["cell_temp"],
            state["current_draw"],
            state["voltage"],
            int(state["charging"])
        ]).reshape(1, -1)
        if not self.trained:
            return "normal"
        return self.model.predict(features)[0]

