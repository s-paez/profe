"""
Central command-line interface for the PROFE pipeline.

Provides a unified interface for all pipeline stages.
"""

import argparse
import sys


def print_manual() -> None:
    """Prints the PROFE manual in terminal."""
    man_text = """\033[1mPROFE MANUAL (Reduction Pipeline for OPTICAM Photometry of Exoplanets)\033[0m

    PROFE is a two-step pipeline for astronomical data reduction.

    \033[1mUSAGE:\033[0m
        profe [OPTIONS]

    \033[1mOPTIONS:\033[0m
        \033[1m-p, --preprocess\033[0m
            Runs the EXOPLANET photometry preprocessing stage.
            This step updates FITS headers (Julian Dates), organizes the
            files by type (Dark, Flat, Science), creates a summary.dat,
            and applies a median filter to the image data.

        \033[1m-n CORES, --cores CORES\033[0m
            Specifies the number of CPU cores to use during preprocessing.
            (e.g., profe -p -n 4). Defaults to all available cores.

        \033[1m-o, --output\033[0m
            Runs the post-processing and output generation stage.
            This utilizes AstroImageJ photometry tables (.tbl) to generate
            Altitude-Azimuth tracks, Binned Light Curves with RMS error,
            Time-averaging Correlated Noise plots, and ExoFOP files.

        \033[1m-h, --help\033[0m
            Shows the quick-reference help message and exits.

        \033[1mman\033[0m
            Displays this detailed manual.

    \033[1mEXAMPLES:\033[0m
        1. Preprocess data using all available processor cores:
           $ profe -p

        2. Preprocess data restricting the process to 4 cores:
           $ profe -p -c 4

        3. Generate scientific outputs and plots from photometric tables:
           $ profe -o
    """
    print(man_text)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].lower() == "man":
        print_manual()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="PROFE: Reduction Pipeline for OPTICAM Photometry of Exoplanets",
        add_help=True,
    )

    # Mutually exclusive group: user can't use -p and -o together
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--preprocess",
        action="store_true",
        help="Run the preprocessing pipeline.",
    )
    group.add_argument(
        "-o",
        "--output",
        action="store_true",
        help="Run the output generation pipeline.",
    )

    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for preprocessing (-p). Defaults to all available.",
    )

    # If no arguments provided, show standard help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # -n is only used with -p
    if args.output and args.cores is not None:
        parser.error("-c/--cores can only be used with the preprocessing (-p) step.")

    if args.preprocess:
        try:
            from profe.preprocess.cli import run_preprocess

            run_preprocess(cores=args.cores)
        except ImportError as e:
            print(f"Error importing preprocessing module: {e}")
            sys.exit(1)

    elif args.output:
        try:
            from profe.output.cli import run_output

            run_output()
        except ImportError as e:
            print(f"Error importing output module: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
