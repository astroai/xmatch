import os
# ...existing imports...

# Get directory where the script is located
testdir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(testdir, "inputs")
output_dir = os.path.join(testdir, "outputs")
os.makedirs(output_dir, exist_ok=True)

# Then replace any instances of "tests/inputs" with input_dir
# and "tests/outputs" with output_dir
