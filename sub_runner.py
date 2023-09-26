import subprocess


if __name__ == '__main__':
    print('Enter script name to run:')
    script_name = input()

    subprocess.Popen(['python3', f'{script_name}.py'])
