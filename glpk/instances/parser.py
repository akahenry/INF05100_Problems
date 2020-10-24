import argparse

parser = argparse.ArgumentParser(
        description='.',
        usage='parser.py [-h] [--input file.col] [--output file.dat]')

parser.add_argument('-i', '--input', type=str, nargs=1,
                    help='a col file containing the instance',
                    required=True)

parser.add_argument('-o', '--output', type=str, nargs=1,
                    help='name of dat file to be created',
                    required=True)

args = parser.parse_args()

if not str(args.input[0]).endswith('.col'):
    print(f'Error: parameter --input must be a .col file ({args.input}).')
    exit(-1)
elif not str(args.output[0]).endswith('.dat'):
    args.output = args.output + '.dat'

input_filename = str(args.input[0])
output_filename = str(args.output[0])

with open(output_filename, 'w') as dat_file:
    dat_file.write('data;\n')
    with open(input_filename) as col_file:
        for line in col_file:
            if line.startswith('c '):
                continue
            elif line.startswith('p '):
                parts = line.replace('\n', '').split(' ')
                number_of_vertices = parts[2]
                
                dat_file.write(f'param V := {number_of_vertices};\n')
                dat_file.write('param E := \n')
            elif line.startswith('e '):
                parts = line.replace('\n', '').split(' ')
                first_vertex = int(parts[1]) - 1
                second_vertex = int(parts[2]) - 1

                dat_file.write(f'{first_vertex} {second_vertex} 1\n')
    dat_file.write(';\n')
    dat_file.write('end;')

        