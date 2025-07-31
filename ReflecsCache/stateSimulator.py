import random
import math

class StateSimulator:
    def __init__(self):
        self.cell_temp = 30.0
        self.current_draw = 70.0
        self.voltage = 400.0
        self.ambient_temp = 25.0
        self.charging = False
        self.power_cut = False
        self.error_rate = 0.05  # %5 hata oranı
        self.error_amount = 0

        self.time_step = 1.0  # saniye
        self.cell_mass = 0.5  # kg
        self.cell_specific_heat = 900  # J/(kg·°C)
        self.cooling_efficiency = 0.5  # soğutmanın etkinliği

        self.action_effect = {
            "cut_power": 0,
            "cool_down": 0,
            "limit_current": 0,
            "increase_current": 0,
        }


    def step(self, action,effect_timer=0):
        YELLOW = '\033[93m'
        PURPLE = '\033[95m'
        RED = '\033[91m'
        RESET = '\033[0m'
        # Aksiyon etkilerini zaman içinde sürdür
        for act in self.action_effect:
            if self.action_effect[act] > 0:
                self.action_effect[act] -= 1


        if self.action_effect["cut_power"] == 0:
            self.power_cut = False
        # Gerçek zamanlı tepki başlat
        self.action_effect[action] = effect_timer



        # Güncel etkileri uygula
        if self.action_effect["cut_power"] > 0:
            self.power_cut = True
            self.current_draw *= 0.8
            self.voltage *= 0.99
            self.cell_temp *= 0.7
        if self.action_effect["cool_down"] > 0:
            # Ortamla ısı farkına göre ekstra soğutma
            if self.cell_temp > 40:
                self.cooling_efficiency *= 1.2
            else:
                self.cooling_efficiency = 0.5
            self.cell_temp -= self.cooling_efficiency * (self.cell_temp - self.ambient_temp)
        if self.action_effect["limit_current"] > 0:
            self.current_draw *= 0.7
        if self.action_effect["increase_current"] > 0:
            self.current_draw *= 1.20



        if not self.power_cut:
            # Arıza ihtimaliyle spike
            if random.random() < self.error_rate:
                self.cell_temp += random.uniform(5, 10)
                print(f"{RED}Spike!!! Arıza: Sıcaklık artışı!{RESET}")
                self.error_amount += 1

            # Joule Heating: I^2 * R
            internal_resistance = 0.01  # ohm
            power_loss = (self.current_draw ** 2) * internal_resistance
            temp_rise = power_loss * self.time_step / (self.cell_mass * self.cell_specific_heat)
            self.cell_temp += temp_rise * 5  # scale-up for simplicity
            print(f"{RED}Joule Heating:{temp_rise*5}{RESET}")

            # Akım ve voltaj değişimleri
            self.voltage += random.uniform(-1, 1)
            self.current_draw += random.uniform(-1, 3)


            # Şarj durumu, ısıyı artırabilir
            if self.charging:
                self.cell_temp += 0.5

            # Aşırı voltaj/düşük voltaj etkisi
            if self.voltage > 450:
                self.current_draw += 5
                print(f"{PURPLE}High Voltage: Current increased!{RESET}")
            elif self.voltage < 320:
                self.current_draw -= 5
                print(f"{PURPLE}Low Voltage: Current decreased!{RESET}")

        # Ortamla doğal ısı dengesi (pasif soğuma)
        passive_cooling = 0.02
        self.cell_temp -= passive_cooling * (self.cell_temp - self.ambient_temp)
        self.ambient_temp += random.uniform(-0.2, 0.2)
        print(f"{YELLOW}Passive Cooling:{passive_cooling * (self.cell_temp - self.ambient_temp)}{RESET}")
        # Clamp değerleri
        self.cell_temp = min(max(self.cell_temp, 0), 80)
        self.current_draw = min(max(self.current_draw, 20), 400)
        self.voltage = min(max(self.voltage, 300), 500)
        self.ambient_temp = min(max(self.ambient_temp, 10), 45)

        # Şarj durumu bazen değişir
        if random.random() < 0.1: # %10 ihtimal
            self.charging = not self.charging
        #Akim arizasi
        if random.random() < self.error_rate: # %5 ihtimal
            value = random.uniform(20, 50)
            self.current_draw += value
            print(f"{PURPLE}AKIM ARTISI: {value}{RESET}")
            self.error_amount += 1

        return {
            "cell_temp": self.cell_temp,
            "current_draw": self.current_draw,
            "voltage": self.voltage,
            "ambient_temp": self.ambient_temp,
            "charging": self.charging
        }
