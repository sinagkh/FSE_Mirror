"""Command-line entry point for the MIRROR artifact."""
import argparse
import runpy
import sys

from mirror.paths import ARTIFACT_ROOT

COMMANDS = {
    "verify": ("verify.py", "Check the artifact's file hashes and study coverage."),
    "test": ("test_core.py", "Test the metamorphic specification compiler."),
    "objective-test": ("test_objectives.py", "Check selected loss values and gradients on CPU."),
    "reproduce": ("reproduce_results.py", "Recompute results and intervals from saved scores."),
    "figures": ("plot_results.py", "Generate the paper's result plots."),
    "download": ("download_assets.py", "Download public assets listed in the configuration."),
    "upstream": ("fetch_upstream.py", "Fetch pinned third-party implementations."),
}


def main():
    parser = argparse.ArgumentParser(prog="python -m mirror")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, (_, description) in COMMANDS.items():
        subparsers.add_parser(command, help=description, add_help=False)
    args, remaining = parser.parse_known_args()
    script = ARTIFACT_ROOT / "scripts" / COMMANDS[args.command][0]
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
