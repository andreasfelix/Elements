import sys, os, ast
import argparse
from . import __version__
from .plotting import plot_lattice
from .latticefile import read_lattice_file_json
from .linbeamdyn import LinBeamDyn

from matplotlib.backends.backend_pdf import PdfPages


def read_lattice(path):
    main_cell = read_lattice_file_json(path)
    lin = LinBeamDyn(main_cell)
    return main_cell, lin


def main():
    parser = argparse.ArgumentParser(description='This is the elements CLI.')
    parser.add_argument('--version', action='version', version=str(__version__))
    subparsers = parser.add_subparsers(dest='subparser')

    subparsers.add_parser('help', help='Get help')

    parser_plot = subparsers.add_parser('plot', help='Plot lattices')
    parser_plot.set_defaults(func=plot)
    parser_plot.add_argument('path', nargs='+', type=str, default='.', help='Path to file or directory.')
    parser_plot.add_argument('-o', '--output_path', type=str, default='out.pdf', help="Output path")
    parser_plot.add_argument('-ref', '--ref_lattice_path', type=str, help="Path to reference lattice")
    parser_plot.add_argument('-ymin', type=int, help="Min Y-value")
    parser_plot.add_argument('-ymax', type=int, help="Max Y-value")
    parser_plot.add_argument('-s', '--sections', type=str, help="Sections")

    args = parser.parse_args()

    if len(sys.argv) < 2 or args.subparser == "help":
        parser.print_help()
    else:
        args.func(args)


def plot(args):
    path_list = args.path
    ref_path = os.path.abspath(args.ref_lattice_path) if args.ref_lattice_path else None
    sections = ast.literal_eval(args.sections) if args.sections else None

    lattice_files = []
    for path in path_list:
        abs_path = os.path.abspath(path)

        if os.path.isfile(abs_path):
            lattice_files.append(abs_path)
        elif os.path.isdir(abs_path):
            for sub_path, sub_dirs, files in os.walk(abs_path):
                files.sort()
                lattice_files.extend([os.path.join(abs_path, sub_path, file) for file in files if file.endswith('.json')])
        else:
            print(f"There is no {abs_path}!")

    with PdfPages(args.output_path) as pdf:
        for file_path in lattice_files:
            print(f"plotting {file_path}")
            main_cell, lin = read_lattice(file_path)
            ref_main_cell = read_lattice_file_json(ref_path) if ref_path else None
            ref_twiss = LinBeamDyn(ref_main_cell).get_twiss() if ref_main_cell else None
            plot_lattice(lin, ref_twiss=ref_twiss, sections=sections, ymin=args.ymin, ymax=args.ymax)
            pdf.savefig()


if __name__ == "__main__":
    main()
