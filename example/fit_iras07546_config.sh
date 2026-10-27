#!/usr/bin/env bash
# Fit IRAS 07546+3928 from fit_iras07546_config.toml -- the file-driven version of
# fit_iras07546_jax.py.
#
#   ./fit_iras07546_config.sh                 # the published fit (~12 min on a GPU, hours on a CPU);
#                                             # a progress bar appears when run in a terminal
#   ./fit_iras07546_config.sh --dry-run       # what would be fitted, without sampling
#   ./fit_iras07546_config.sh --device cpu    # force the CPU
#   ./fit_iras07546_config.sh --output /tmp/x # ... or any other option of the CLI
#
# Everything -- model, priors, sampler settings, output location -- is in the .toml
# file; whatever is given after the script name is passed on to
# `python -m galspec.pipeline` and overrides it.  The paths inside the configuration
# are resolved relative to the configuration, so this script works from any
# directory; the outputs land under example/out/iras07546.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

args=("$@")
if [[ " $* " != *" --device "* ]]; then
    args+=("--device" "${GALSPEC_DEVICE:-auto}")
fi

exec python -m galspec.pipeline "$here/fit_iras07546_config.toml" ${args[@]+"${args[@]}"}