class ReflexNetGenerator:
    def __init__(self):
        self.last_action = None
        self.action_timer = 0

    def predict(self, state):
        temp = state["cell_temp"]
        current = state["current_draw"]
        voltage = state["voltage"]
        ambient = state["ambient_temp"]
        charging = state["charging"]

        # Eğer önceki aksiyon halen etkiliyse, devam ettir
        if self.action_timer > 0:
            self.action_timer -= 1
            return "normal"

        # Yeni aksiyon kontrolü
        if temp > 55 or (charging and temp > 95):
            self.last_action = "cut_power"
            self.action_timer = 3
        elif current > 200:
            self.last_action = "limit_current"
            self.action_timer = 1
        elif current < 100:
            self.last_action = "increase_current"
            self.action_timer = 1
        elif temp > 30 or current > 150 :
            self.last_action = "cool_down"
            self.action_timer = 2
        elif temp > 40 and ambient > 35:
            self.last_action = "cool_down"
            self.action_timer = 2
        else:
            self.last_action = "normal"
            self.action_timer = 0

        return self.last_action
