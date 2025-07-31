from ai_memory import AIMemory
from env_simulator import Environment
from input_generator import generate_random_input
from memory import ReflexMemory
from controller import should_stop
from logger import print_step_log
from colorama import Fore, Style, init

init(autoreset=True)
ai_memory = AIMemory()
env = Environment()
memory = ReflexMemory()
decision = "CONTINUE"

for step in range(1, 50):

    if decision == "STOP":
        env.stabilization()
        state = env.get_state()
        action = {"type": "Stabilization"}
        from_memory = False
        print_step_log(memory.get_thresholds(),step, action, state, decision, from_memory, color=Fore.BLUE)
        decision = "STOP" if should_stop(state, memory.get_thresholds()) else "CONTINUE"
        continue
    else:
        action = generate_random_input()
        # Refleks kontrolü
        if memory.is_known_action(action):
            decision = "STOP"
            from_memory = False
        else:
            env.apply_action(action)
            state = env.get_state()
            # --- AI tahmini ---
            ai_decision = ai_memory.predict(action, state)

            if ai_decision == 1:
                decision = "STOP"
                from_memory = True
            else:
                decision = "STOP" if should_stop(state, memory.get_thresholds()) else "CONTINUE"
                from_memory = False

            is_anomaly = (decision == "STOP")
            ai_memory.record(action, state, is_anomaly)
            # Eğer kritik durum varsa öğren
            if is_anomaly:
                memory.record_anomaly(action, state)

                print_step_log(memory.get_thresholds(), step, action, state, decision, from_memory, color=Fore.RED)

            else:
                memory.record_normal(action, state)
                print_step_log(memory.get_thresholds(), step, action, state, decision, from_memory, color=Fore.GREEN)

        if from_memory:
            print_step_log(memory.get_thresholds(), step, action, state, decision, from_memory,
                           color=Fore.YELLOW)

