import numpy as np
from sklearn.linear_model import LogisticRegression

class AIMemory:
    def __init__(self):
        self.X = []
        self.y = []
        self.model = None
        self.trained = False
        self.action_keys = None
        self.state_keys = None
        self.value_maps = {}  # String -> int dönüşümü için

    def _encode(self, key, value):
        # Eğer değer string ise, sayıya çevir
        if isinstance(value, str):
            if key not in self.value_maps:
                self.value_maps[key] = {}
            if value not in self.value_maps[key]:
                self.value_maps[key][value] = len(self.value_maps[key])
            return self.value_maps[key][value]
        return value

    def record(self, action, state, is_anomaly):
        if self.action_keys is None:
            self.action_keys = list(action.keys())
        if self.state_keys is None:
            self.state_keys = list(state.keys())
        features = [self._encode(f"action_{k}", action.get(k, 0)) for k in self.action_keys] + \
                   [self._encode(f"state_{k}", state.get(k, 0)) for k in self.state_keys]
        self.X.append(features)
        self.y.append(1 if is_anomaly else 0)
        if len(self.X) >= 30 and not self.trained:
            self.train_model()

    def train_model(self):
        self.model = LogisticRegression()
        self.model.fit(np.array(self.X), np.array(self.y))
        self.trained = True

    def predict(self, action, state):
        if not self.trained:
            return None
        features = [self._encode(f"action_{k}", action.get(k, 0)) for k in self.action_keys] + \
                   [self._encode(f"state_{k}", state.get(k, 0)) for k in self.state_keys]
        features = np.array([features])
        return self.model.predict(features)[0]