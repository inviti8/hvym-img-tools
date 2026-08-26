# hvym-img-tools — Agent Onboarding & Architecture

Everything a new agent needs to start contributing. **Read this first.**

---

## TL;DR

- **What:** a suite of **self-hosted, AI-augmented image tools** exposed as a **GPU
  service** (+ CLIs). First client: **Inkternity** (a C++ drawing app). First tool:
  **reangle** (style-preserving camera-angle adjustment of a drawing).
- **Principle:** **sovereign** — everything self-hostable, no external AI APIs,
  permissive licenses preferred. And **modular** — reangle is tool #1 of N; the framework
  is built so tools 2..N slot in without touching each other.
- **Shape:** a shared **core** (server, model cache, image/3D utils) + pluggable **tools**
  (each a self-contained package implementing one `Tool` interface). One GPU container
  serves all tools.
- **Stack:** Python, **managed with `uv`** (never raw pip/python), FastAPI, torch/CUDA.
- **The reangle spec is authoritative in Inkternity's repo:**
  `../infinipaint/docs/design/REANGLE_PIPELINE.md` (this repo implements it).

---

## 1. Vision & scope

Creative apps increasingly want AI *augmentation* that respects the artist — non-invasive,
style-preserving, and not dependent on a third-party cloud. `hvym-img-tools` is where those
capabilities live: a growing set of image/2.5D/3D tools, each self-hosted, each with a
clean HTTP contract a native app can call.

- **First tool — reangle:** take a character drawing, build a rough 3D proxy, and let the
  artist turn the camera a little while **preserving their exact linework** (the drawing is
  re-projected, never regenerated). Proven end-to-end (see §5).
- **Design for more:** relighting, style-consistent inpainting/outpainting, turnaround
  sheets, depth/normal extraction, texture fill — all fit the same tool model. **Build
  reangle as a well-factored example, not a one-off.**

Non-goals: not a general model zoo, not a hosted SaaS wrapper. Sovereign, app-facing tools.

---

## 2. Architecture (modular)

```
                      ┌─────────────────────────────────────────────┐
   client (Inkternity)│                 core.server                  │
   ─── HTTP ─────────▶│  FastAPI app; mounts every registered tool   │
                      │  at POST /tools/{name}, aggregates OpenAPI    │
                      └───────┬───────────────────────┬──────────────┘
                              │                        │
                     core.registry            core.models (ModelCache)
                    (discovers tools)        (warm, lazy, shared GPU models)
                              │                        │
                    ┌─────────▼─────────┐    ┌──────────▼───────────┐
                    │  tools/reangle    │    │  tools/<next>        │
                    │  Tool: schemas +  │    │  Tool: schemas +     │
                    │  run() pipeline   │    │  run() pipeline      │
                    └───────────────────┘    └──────────────────────┘
```

- **core** — the framework every tool reuses: the FastAPI server, a **tool registry**
  (discovery + mounting), a **model cache** (load heavy models once, share across tools and
  requests), image I/O + matting helpers, a **result cache keyed by input hash**, config.
- **tools/** — one package per capability. A tool declares its name, typed input/output,
  which models it needs warmed, and a `run()`. It knows nothing about other tools.
- **one process, one container** — all tools share the GPU and warm models; the server
  routes by tool name. Deploy the same image to RunPod serverless or a persistent box.

This is the "modular thinking": **adding a tool = adding a package under `tools/` and
registering it.** No edits to core, no edits to other tools.

---

## 3. Proposed repo layout

```
hvym-img-tools/
├── AGENTS.md                     # this file — start here
├── README.md                     # short human overview
├── pyproject.toml                # uv-managed deps
├── docker/Dockerfile             # GPU image (see §6)
├── docs/
│   └── tools/reangle.md          # per-tool notes (or link to inkternity's REANGLE_PIPELINE.md)
├── hvym_img_tools/
│   ├── core/
│   │   ├── tool.py               # Tool ABC (the contract, §4)
│   │   ├── registry.py           # register() + discovery, mount into FastAPI
│   │   ├── server.py             # build the FastAPI app from the registry
│   │   ├── models.py             # ModelCache — warm/lazy/shared model loading
│   │   ├── imageio.py            # decode/encode, isnet matte, resize/pad
│   │   ├── cache.py              # artifact cache keyed by sha256(input+params)
│   │   └── config.py
│   ├── tools/
│   │   └── reangle/
│   │       ├── __init__.py       # register(ReangleTool)
│   │       ├── tool.py           # ReangleTool(Tool): I/O schemas + run()
│   │       ├── pipeline.py       # matte → reconstruct → uv-bake → glb
│   │       ├── reconstruct.py    # SWAPPABLE backbone (TripoSR / DrawingSpinUp)
│   │       └── uvbake.py         # front-projection UV atlas → embedded-texture glb
│   └── cli.py                    # `hvym-img <tool> --in ... --out ...`
├── scripts/                      # dev/deploy helpers (runpod spin-up, smoke tests)
└── tests/                        # core + per-tool tests (CPU-only where possible)
```

Not prescriptive to the letter — but keep the **core / tools split** and the **Tool
contract** intact.

---

## 4. The `Tool` contract (how modularity works)

Every tool implements one small interface. Illustrative sketch (finalize in `core/tool.py`):

```python
class Tool(ABC):
    name: str                       # url slug, e.g. "reangle"
    summary: str
    version: str = "0.1.0"
    InputModel: type[BaseModel]     # pydantic; may include file fields (multipart)
    # Output is either a pydantic model (JSON) or a MediaResponse (bytes + media type),
    # e.g. a .glb or .png. Declare which.

    def models_needed(self) -> list[str]: ...      # keys the ModelCache should warm
    def run(self, req: "InputModel", ctx: Context) -> "OutputModel | MediaResponse": ...
```

- **`core.registry`** collects every `Tool` and mounts it at `POST /tools/{name}`, feeding
  its `InputModel`/`OutputModel` into FastAPI so **OpenAPI is generated automatically**.
- **`core.models.ModelCache`** loads each needed model **once at startup** (warm) and hands
  tools a ready handle in `ctx` — no per-request reloads.
- **`core.cache`** memoizes expensive results by `sha256(input + params)` — e.g. reangle's
  reconstructed mesh, so a re-request is instant.
- `ctx: Context` carries the model cache, the result cache, a temp workspace, and config.

That's the whole framework surface a tool touches.

---

## 5. Reangle — the reference tool (tool #1)

**What it does.** Drawing in → a **textured `.glb`** out: a rough 3D proxy of the character
with the artist's **original art front-projected** as its texture. The client (Inkternity)
loads it into its 3D viewer, orbits the camera a little, and bakes the view to canvas.
Style is preserved because we move the real pixels, never regenerate them.

**Pipeline** (`tools/reangle/pipeline.py`):
1. **matte** — isnet → 512² RGBA (character on transparent). `core.imageio`.
2. **reconstruct** — single-image → 3D mesh. **Swappable backbone** (`reconstruct.py`):
   prefer **TripoSR (MIT, ~seconds, outputs a mesh directly)**; the validated prototype
   used **DrawingSpinUp (Wonder3D+NeuS)** — but **Wonder3D weights are CC-BY-NC → demo
   only**, so TripoSR is the productization target.
3. **uv-bake** — front-planar UV (`uv = normalize(vertex.xy)`) with the original art as the
   texture (later: xatlas unwrap + disocclusion inpainting for wider angles). `uvbake.py`.
4. **export** — embedded-texture `.glb` (cgltf/tinygltf/trimesh).

**Endpoint:** `POST /tools/reangle` — multipart image in → `.glb` out. Mesh cached by input
hash (§4).

**Status:** the *pipeline is proven* end-to-end (sovereign, on a RunPod A6000) — see the
authoritative spec and findings in **`../infinipaint/docs/design/REANGLE_PIPELINE.md`** and
the prototype scripts in **`../infinipaint/scripts/reangle/`** (matte, depth-warp, mesh
render, and the negative-result experiments). This repo's job is to turn that prototype into
a clean, deployable `Tool`.

**Hard-won findings (don't re-litigate — full detail in REANGLE_PIPELINE.md):**
- Generative reangle drifts style (baked into weights) — rejected.
- Monocular depth (Depth-Anything) **fails** — it gives scene depth, not object relief.
- **Pose is the constraint**, not medium: thin protrusions (outstretched arms, frilly
  skirts) collapse the mesh; neutral stances reconstruct cleanly.
- The **client** integration is the **Inkternity armature 3D viewer** + bake (not a
  server-side per-angle slider) — see REANGLE_PIPELINE.md §7.

---

## 6. Environment & running

- **Python via `uv`** (repo convention — every invocation goes through uv, never raw
  pip/python).
- **GPU deps live in `docker/Dockerfile`.** The full, battle-tested environment recipe
  (CUDA 11.8 toolchain, torch, pytorch3d, model downloads, and every gotcha) is in
  `../infinipaint/docs/design/REANGLE_PIPELINE.md` §5 — encode it in the Dockerfile.
  **Note:** switching the backbone to **TripoSR** drops tiny-cuda-nn / NeuS / Wonder3D and
  massively simplifies this image — do that early.
- Keep **`core` importable CPU-side** (no GPU import at module load) so tests and the
  registry run without a GPU.
- **Run (target):** `uv run hvym-img-serve` → FastAPI on a port; `POST /tools/reangle`.
  Deploy the container to **RunPod serverless** (scale-to-zero) or a persistent 24 GB box.
- **CLI (target):** `uv run hvym-img reangle --in drawing.png --out char.glb`.

---

## 7. Adding a new tool (the modular workflow)

1. `hvym_img_tools/tools/<name>/` — new package.
2. Implement `class <Name>Tool(Tool)`: set `name`, `summary`, `InputModel`,
   `OutputModel` (or media), `models_needed()`, and `run()`.
3. `register(<Name>Tool)` in the package `__init__.py`.
4. Put heavy model loads behind `ModelCache` (declare them in `models_needed()`); use
   `core.cache` for expensive intermediate artifacts.
5. Add `docs/tools/<name>.md` + tests. Done — it auto-mounts at `POST /tools/<name>` with
   generated OpenAPI. **No core or sibling-tool edits.**

Checklist for a good tool: sovereign (self-hostable weights), permissive license (flag any
CC-BY-NC as demo-only in its doc), CPU-importable module, typed I/O, cached heavy steps.

---

## 8. API conventions

- **Route:** `POST /tools/{name}`; `GET /tools` lists tools + versions; `GET /healthz`.
- **I/O:** pydantic models → auto OpenAPI at `/docs`. Binary outputs (`.glb`, `.png`)
  returned as a media response with the right content-type.
- **Idempotency/caching:** cache by `sha256(input bytes + params)`; a repeat call is instant.
- **Long jobs:** if a tool exceeds a few seconds cold, support an async job form
  (`202 + job id`, `GET /jobs/{id}`) — reangle's first reconstruction is the case to design
  for; cached re-requests are fast.
- **Versioning:** per-tool `version`; breaking contract changes bump it. Keep the contract
  in sync with any client (for reangle, that's REANGLE_PIPELINE.md §7.5).

---

## 9. Conventions & constraints

- **Sovereign first.** Self-hostable weights only; no external AI API calls from tools.
- **License hygiene.** Prefer MIT/Apache. **Document each tool's model licenses**; anything
  CC-BY-NC (e.g. Wonder3D) is **demo/research only** — don't let it into a shipped default.
- **`uv` for everything Python.**
- **Keep core generic** — no tool-specific logic in `core`. If two tools need the same
  helper, it goes in `core`, not copied.
- **Small, typed contracts** at the tool boundary; big/ugly ML internals stay inside the
  tool.

---

## 10. Context pointers (sibling repo & prototype)

- **Inkternity repo:** `../infinipaint` (product name "inkternity", C++23). It is the first
  **client**, not part of this repo.
  - `docs/design/REANGLE_PIPELINE.md` — **the authoritative reangle spec** (pipeline, env
    recipe, viewer-based integration, licensing). Source of truth; keep in sync.
  - `docs/design/AI_CAMERA_ANGLE_ADJUST.md` — the research + why generative/monocular were
    rejected.
  - `scripts/reangle/` — the working prototype scripts to port from (`prep_input.py`,
    `depthwarp.py`, `render_p3d.py`, + experiments).
- **What's proven vs. what's next:** the *pipeline* is proven; this repo builds the *server*
  (port pipeline → `ReangleTool`, add TripoSR backbone, UV-bake, glb export, mesh cache).

---

## 11. Roadmap

1. **Scaffold core** — `Tool` ABC, registry, server, ModelCache, cache, imageio.
2. **Reangle tool** — port the prototype; add the **front-projection UV-bake → glb** and the
   `POST /tools/reangle` endpoint; cache the mesh by input hash.
3. **TripoSR backbone** — replace DrawingSpinUp/Wonder3D (MIT, faster, lighter image).
4. **Dockerfile + deploy** — RunPod; smoke-test end to end.
5. **Inkternity client** — HTTP client + armature-viewer textured render + bake
   (REANGLE_PIPELINE.md §7).
6. **Tool #2** — pick the next AI-augmented image capability and prove the framework
   generalizes.
