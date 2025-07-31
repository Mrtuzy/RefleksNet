from colorama import Fore
def print_step_log(trasholds,step, action, state, decision, from_memory, color=Fore.GREEN):
    print(color + f"[{step}] Action={action} | State={state} | Decision={decision} | Reflex={from_memory}")
    print("Trashold:", trasholds)
