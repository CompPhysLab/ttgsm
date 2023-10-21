import subprocess


if __name__ == '__main__':
    # print('Enter script name to run:')
    # script_name = input()
    script_name = 'multiscale_convergence'

    print('n range:', end='\t')
    n_range_str = input()

    print('d range from:', end='\t')
    d_from_str = input()
    print('d range to:', end='\t')
    d_to_str = input()

    print('thickness:', end='\t')
    thickness_str = input()

    print('accuracy:', end='\t')
    accuracy_str = input()

    print('num sweeps:', end='\t')
    nswp_str = input()

    subprocess.Popen(['python3', f'{script_name}.py', n_range_str, d_from_str, d_to_str, thickness_str, accuracy_str, nswp_str])
