import re
import os

def update_stilts():
    filepath = "src/xmatch/stilts.py"
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    changed = False

    for line in lines:
        if 'df_to_use = df.copy()' in line:
            new_lines.append(line.replace('df.copy()', 'df.copy(deep=False)  # ⚡ Bolt: Use shallow copy for memory optimization before possible string conversion'))
            changed = True
        else:
            new_lines.append(line)

    if changed:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"Updated {filepath}")

def update_astro_utils():
    filepath = "src/xmatch/astro_utils.py"
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    changed = False

    for line in lines:
        if 'df_out = df.copy()' in line:
            new_lines.append(line.replace('df.copy()', 'df.copy()  # Deep copy is kept here to avoid unexpected mutation of caller DataFrame with .loc/.iloc assignments'))
        elif 'df_filled = df[cols_for_prop].copy()' in line:
            new_lines.append(line.replace('df[cols_for_prop].copy()', 'df[cols_for_prop].copy()  # Deep copy is required here before doing .loc assignments with NaN masking to avoid modifying original dataframe'))
        elif 'df_work = df.copy()' in line:
            new_lines.append(line.replace('df.copy()', 'df.copy()  # Deep copy is kept because we append column in-place if missing'))
        else:
            new_lines.append(line)

    if changed:
        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"Updated {filepath}")

update_stilts()
