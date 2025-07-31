import numpy as np
class ReflexMemory:
    def __init__(self):
        self.records = []  # Liste olarak kaydedilecek
        self.learned_actions = {}  # input signature → refleks kararı
        self.thresholds = {
            "vibration": None,
            "stress": None,
            "noise": None,
        }

    def record_anomaly(self, action, state):
        self.records.append({"action": action, "state": state, "is_anomaly": True})
        self.learned_actions[str(action)] = "STOP"
        self.update_thresholds()

    def record_normal(self, action, state):
        self.records.append({"action": action, "state": state, "is_anomaly": False})
        self.update_thresholds()

    def update_thresholds(self):
        for key in self.thresholds:
            # Sadece anomali olmayan kayıtları al
            values = [r["state"][key] for r in self.records ]
            if len(values) < 4:
                continue  # Yeterli veri yoksa güncelleme
            median = np.median(values)
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            self.thresholds[key] = median + 1.5 * iqr

    def is_known_action(self, action):
        return str(action) in self.learned_actions

    def get_thresholds(self):
        return self.thresholds
