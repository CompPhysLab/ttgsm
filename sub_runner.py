import subprocess


print('Enter script name to run:')
scriptname = input()

subprocess.Popen(['python3', f'{scriptname}.py'])

