"""Lazy phase dispatcher keeping evaluator imports out of decision processes."""

import sys

from socratic_tutor.benchmark.command_io import print_command_error, print_command_result


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"decision", "criterion"}:
        error = ValueError("benchmark-generate requires phase: decision or criterion")
        print_command_error(command="generate", error=error)
        return 2
    phase, phase_arguments = arguments[0], arguments[1:]
    try:
        if phase == "decision":
            result = _run_decision(phase_arguments)
            leaked = tuple(
                name
                for name in sys.modules
                if name.startswith("socratic_tutor.benchmark.evaluator")
            )
            if leaked:
                raise RuntimeError("Decision process imported evaluator benchmark modules")
        else:
            result = _run_criterion(phase_arguments)
    except Exception as error:
        print_command_error(command=f"generate-{phase}", error=error)
        return 1
    print_command_result(command=f"generate-{phase}", result=result)
    return 0


def _run_decision(argv: list[str]):
    from socratic_tutor.benchmark.public.decision_command import run_decision_command

    return run_decision_command(argv)


def _run_criterion(argv: list[str]):
    from socratic_tutor.benchmark.evaluator.criterion_command import run_criterion_command

    return run_criterion_command(argv)
