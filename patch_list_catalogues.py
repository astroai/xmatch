import sys

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Revert list_catalogues error path
    content = content.replace(
        'print("No catalogues configured.", file=sys.stderr)\n        return False',
        'print("No catalogues configured.", file=sys.stderr)\n        return'
    )

    with open(filepath, 'w') as f:
        f.write(content)

patch_file('src/xmatch/cli.py')
