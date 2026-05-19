"""
Central command-line interface for the PROFE pipeline.

Provides a unified interface for all pipeline stages.
"""

import argparse
import sys

from profe.logger import setup_logging


def print_manual() -> None:
    """Prints the PROFE manual in terminal."""
    man_text = """\033[1mPROFE MANUAL (Reduction Pipeline for OPTICAM Photometry of Exoplanets)\033[0m

    PROFE is a two-step pipeline for astronomical data reduction.

    \033[1mUSAGE:\033[0m
        profe [OPTIONS]

    \033[1mOPTIONS:\033[0m
        \033[1m-p, --preprocess\033[0m
            Runs the entire preprocessing pipeline.
            This step updates FITS headers (Julian Dates), organizes the
            files by type (Dark, Flat, Science), creates a summary.dat,
            and applies a median filter to the image data.

        \033[1m--organice\033[0m
            Runs ONLY the reorganization stage. Updates FITS headers (JD and
            UTMIDDLE), organizes files, and creates summary.dat.

        \033[1m--filter\033[0m
            Runs ONLY the median filter stage. Skips processing if the
            corrected output directory already exists.

        \033[1m-c CORES, --cores CORES\033[0m
            Specifies the number of CPU cores to use during preprocessing.
            (e.g., profe -p -c 4). Defaults to all available cores.

        \033[1m-o [TARGET], --output [TARGET]\033[0m
            Runs the post-processing and output generation stage.
            This utilizes AstroImageJ photometry tables (.tbl) to generate
            all scientific products: Alt-Az trajectory plots, multi-band
            light curves (PDF/PNG/CSV), time-averaging noise plots,
            field-of-view aperture plots, seeing profiles, comparison star
            curves, WCS-solved FITS, transit timing data from TTF, and
            consolidated ExoFOP notes reports.
            Optionally specify a TARGET name (e.g., TOI-1234) to generate
            outputs only for that target.

        \033[1m-pu [TARGETS], --prepare-upload [TARGETS]\033[0m
            Prepares and packages data products into an intermediate .tar file
            for ExoFOP upload. Excludes unnecessary files and prompts for a
            Data Tag. Optionally specify target(s) to prepare.

        \033[1m-u [TARGETS], --upload [TARGETS]\033[0m
            Uploads the prepared files to ExoFOP iteratively via the single-file
            upload endpoint to preserve original scientific names, using
            credentials stored in the .exofop_credentials file.

        \033[1m-h, --help\033[0m
            Shows the quick-reference help message and exits.

        \033[1mman\033[0m
            Displays this detailed manual.

    \033[1mEXAMPLES:\033[0m
        1. Preprocess data using all available processor cores:
           $ profe -p

        2. Run ONLY the reorganization and header update:
           $ profe --organice

        3. Run ONLY the median filter step:
           $ profe --filter

        4. Preprocess data restricting the process to 4 cores:
           $ profe -p -c 4

        5. Generate scientific outputs and plots from photometric tables:
           $ profe -o

        6. Generate outputs only for a specific target:
           $ profe -o "TOI-1234"

        7. Prepare an ExoFOP upload for multiple targets:
           $ profe --prepare-upload TOI-1234 TOI-5678

        8. Upload all prepared ExoFOP packages:
           $ profe --upload
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

    # Mutually exclusive group: user can't use -p, -o, --organice, or --filter together
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--preprocess",
        action="store_true",
        help="Run the entire preprocessing pipeline (organize + filter).",
    )
    group.add_argument(
        "--organice",
        action="store_true",
        help="Run only the reorganization and JD/UTMIDDLE update stage.",
    )
    group.add_argument(
        "--filter",
        action="store_true",
        help="Run only the median filter stage.",
    )
    group.add_argument(
        "-o",
        "--output",
        nargs="?",
        const=True,
        default=False,
        metavar="TARGET",
        help="Run the output generation pipeline. Optionally specify a target name to process only that target.",
    )
    group.add_argument(
        "-pu",
        "--prepare-upload",
        nargs="*",
        metavar="TARGET",
        help="Prepare and package data products into a .tar file for ExoFOP upload.",
    )
    group.add_argument(
        "-u",
        "--upload",
        nargs="*",
        metavar="TARGET",
        help="Upload the previously prepared .tar files to ExoFOP.",
    )

    parser.add_argument(
        "-c",
        "--cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for preprocessing. Defaults to all available.",
    )

    # If no arguments provided, show standard help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # -c is only used with preprocessing steps
    if args.output and args.cores is not None:
        parser.error(
            "-c/--cores can only be used with preprocessing (-p, --organice, --filter) steps."
        )

    if args.preprocess or args.organice or args.filter:
        # Determine command name for logging
        cmd_name = "preprocess"
        if args.organice:
            cmd_name = "organize"
        elif args.filter:
            cmd_name = "filter"

        setup_logging(cmd_name)

        try:
            from profe.preprocess.cli import run_preprocess

            # Logic:
            # -p (preprocess): do_organize=True, do_filter=True
            # --organice: do_organize=True, do_filter=False
            # --filter: do_organize=False, do_filter=True

            do_organize = args.preprocess or args.organice
            do_filter = args.preprocess or args.filter

            run_preprocess(
                cores=args.cores, do_organize=do_organize, do_filter=do_filter
            )
        except ImportError as e:
            print(f"Error importing preprocessing module: {e}")
            sys.exit(1)

    elif args.output:
        setup_logging("output")

        try:
            from profe.output.cli import run_output

            # args.output is True (bare -o) or a string (target name)
            target = args.output if isinstance(args.output, str) else None
            run_output(target=target)
        except ImportError as e:
            print(f"Error importing output module: {e}")
            sys.exit(1)

    elif args.prepare_upload is not None:
        setup_logging("prepare_upload")
        try:
            from profe.upload.cli import run_prepare_upload
            run_prepare_upload(targets=args.prepare_upload)
        except ImportError as e:
            print(f"Error importing upload module: {e}")
            sys.exit(1)

    elif args.upload is not None:
        setup_logging("upload")
        try:
            from profe.upload.cli import run_upload
            run_upload(targets=args.upload)
        except ImportError as e:
            print(f"Error importing upload module: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
