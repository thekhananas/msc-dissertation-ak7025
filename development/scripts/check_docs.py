"""Compile repository D2 blocks and validate Markdown fence balance."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = WORKSPACE_ROOT.parent
DOCS_ROOT = REPOSITORY_ROOT / "docs"
D2_BLOCK = re.compile(r"^```d2\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def main() -> None:
    d2_binary = shutil.which("d2")
    if d2_binary is None:
        raise SystemExit("D2 is required for docs-check but was not found on PATH")

    diagram_count = 0
    with tempfile.TemporaryDirectory(prefix="socratic-tutor-d2-") as temp_directory:
        output_root = Path(temp_directory)
        for document in sorted(DOCS_ROOT.glob("*.md")):
            content = document.read_text(encoding="utf-8")
            if content.count("```") % 2 != 0:
                raise SystemExit(f"Unbalanced Markdown fences: {document}")
            for block_index, source in enumerate(D2_BLOCK.findall(content), start=1):
                diagram_count += 1
                input_path = output_root / f"{document.stem}-{block_index}.d2"
                output_path = input_path.with_suffix(".svg")
                input_path.write_text(source, encoding="utf-8")
                subprocess.run(
                    [d2_binary, "--layout=dagre", str(input_path), str(output_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

    print(f"Compiled {diagram_count} D2 diagrams from {DOCS_ROOT}")


if __name__ == "__main__":
    main()
