#!/usr/bin/env bash
# One image, three roles. Which one is decided at run time so the same artifact
# can be a serverless worker, a plain HTTP server, or a one-off CLI run.
#
#   serverless  RunPod Serverless worker (default; queue-driven, no HTTP)
#   serve       FastAPI server on $HVYM_PORT (persistent pod / local docker run)
#   cli         `hvym-img ...`, e.g. docker run IMAGE cli reangle --in a.png --out a.glb
#   <anything>  executed verbatim, so `docker run IMAGE bash` still works
set -euo pipefail

MODE="${1:-serverless}"
shift || true

# torch ships its own CUDA libs; onnxruntime-gpu needs cuDNN 9 on the path or it
# silently falls back to CPU (6.4s vs 0.03s per matte). Resolve it at run time
# rather than hard-coding a site-packages path that a rebuild could move.
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
export LD_LIBRARY_PATH="${SITE}/nvidia/cudnn/lib:${SITE}/nvidia/cublas/lib:${SITE}/nvidia/cufft/lib:${SITE}/nvidia/curand/lib:${LD_LIBRARY_PATH:-}"

case "$MODE" in
  serverless)
    exec python -m hvym_img_tools.serverless
    ;;
  serve)
    exec python -m uvicorn hvym_img_tools.core.server:create_app --factory \
         --host "${HVYM_HOST:-0.0.0.0}" --port "${HVYM_PORT:-8000}"
    ;;
  proxy)
    # The proxy needs no GPU; normally run from a separate, cheaper image.
    exec python -m hvym_img_tools.proxy
    ;;
  cli)
    exec python -m hvym_img_tools.cli "$@"
    ;;
  *)
    exec "$MODE" "$@"
    ;;
esac
