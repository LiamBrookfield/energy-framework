#User input prompt, returns algo (str) n, reps, and sudo (bool)

ALGO_MAP = {
	"1": "insertion"
}

def prompt_algo():

	print("""Select Algorithm \n
	1. Insertion
	2. Somethin Else
	3. Another Thing"""
	)

	while True:
		choice = input().strip()
		if choice in ALGO_MAP:
			return ALGO_MAP[choice]
		print("Invalid Choice dickhead, Please enter 1")

# Input function for ints, checks type and value
def prompt_int(message, min_val, max_val):

	while True:
		value = input(message).strip()

		if not value.isdigit():
			print("Enter a valid integer.")
			continue

		value = int(value)

		if min_val is not None and value < min_val:
			print(f"Value must be >= {min_val}")
			continue

		if max_val is not None and value > max_val:
			print(f"Warning: Number too high, enter value below {max_val}")
			continue

		return value

# Ask if user wants to append sudo to the start of the command (usually yes)
def prompt_yes_no(message):
	while True:
        	ans = input(message + " (y/n): ").strip().lower()
        	if ans in ("y", "yes"):
            		return True
        	if ans in ("n", "no"):
            		return False
        	print("Please enter y or n.")

# Put these commands together, returns algo: str, n: int, reps: int, use_sudo: bool
def get_user_inputs():
    algo = prompt_algo()
    n = prompt_int("Enter size n: ", min_val=1, max_val=1000000)
    reps = prompt_int("Enter repetitions: ", min_val=1, max_val=100)
    use_sudo = prompt_yes_no("Run perf with sudo?")
    return algo, n, reps, use_sudo

print(get_user_inputs())
