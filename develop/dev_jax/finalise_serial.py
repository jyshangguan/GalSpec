"""Fill the measured single-core number into the report text, the README and the docs.

Run after ``sdss_dynesty-serial.json`` exists: it rewrites the three places that quote
the single-core timing, so no hand-editing is needed and the numbers cannot drift.
"""
from __future__ import annotations

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "develop" / "dev_jax" / "results"
GPU_MIN = 6.3


def main() -> int:
    rec = json.loads((RESULTS / "sdss_dynesty-serial.json").read_text())
    hours = rec["runtime_seconds"] / 3600.0
    speed = hours * 60.0 / GPU_MIN
    calls = rec["ncall"]
    print(f"single core: {hours:.2f} h, {calls:,} calls, logZ "
          f"{rec['log_evidence']:.2f} +- {rec['log_evidence_error']:.2f}, "
          f"chi2/dof {rec['reduced_chi2']:.2f}")
    print(f"speed-up over one core: {speed:.1f}x")

    readme = ROOT / "develop" / "dev_jax" / "README.md"
    s = readme.read_text()
    s = re.sub(r"\| dynesty, 1 core \|.*?\| n/a \|",
               f"| dynesty, 1 core | {hours:.1f} h (measured) | "
               f"{calls / rec['runtime_seconds']:.0f} |", s)
    s = re.sub(r"\* 13\.3x faster than 16 CPU cores, at least 58x faster than one core",
               f"* 13.3x faster than 16 CPU cores, {speed:.0f}x faster than one core", s)
    s = s.replace("(that run was still going after 6.1 h when this was written).",
                  f"(measured: {hours:.1f} h on one core).")
    readme.write_text(s)

    docs = ROOT / "docs" / "source" / "jax_gpu.rst"
    s = docs.read_text()
    s = s.replace("""   * - dynesty, 1 CPU core
     - > 6.1 h, not finished
     - not finished
     - not finished""",
                  f"""   * - dynesty, 1 CPU core
     - {hours:.1f} h ({rec['runtime_seconds']:.0f} s)
     - {calls / 1e6:.2f} x 10^6
     - {rec['log_evidence']:.1f} +/- {rec['log_evidence_error']:.1f}""")
    docs.write_text(s)
    print("README and docs updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
