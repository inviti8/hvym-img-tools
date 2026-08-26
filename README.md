# hvym-img-tools

A suite of **self-hosted, AI-augmented image tools** for creative apps — served over a
clean HTTP contract (and CLIs) so a native app can call them without depending on any
third-party AI cloud. **Sovereign** (self-hostable, permissive licenses) and **modular**
(each capability is a self-contained *tool* that plugs into a shared core).

**First client:** [Inkternity](../infinipaint) (a C++ drawing app).
**First tool:** **reangle** — style-preserving camera-angle adjustment of a character
drawing (build a rough 3D proxy, re-project the artist's *original* linework, turn the
camera a little; the drawing is moved, never regenerated).

## Architecture at a glance

- **core** — FastAPI server, tool **registry** (auto-mounts tools at `POST /tools/{name}`),
  a shared **model cache** (warm GPU models once), image/3D utilities, input-hash result
  cache.
- **tools/** — one package per capability implementing a small `Tool` interface. Adding a
  tool touches nothing else.
- **one GPU container** serves every tool; deploy to RunPod serverless or a persistent box.

## Start here

**→ [`AGENTS.md`](AGENTS.md)** — full onboarding & architecture: the module model, the
`Tool` contract, the reangle reference tool, environment/Docker, how to add a tool, API
conventions, and the roadmap.

Reangle's authoritative pipeline spec lives in the client repo:
[`../infinipaint/docs/design/REANGLE_PIPELINE.md`](../infinipaint/docs/design/REANGLE_PIPELINE.md).

## Conventions

- Python via **`uv`** (never raw pip/python).
- Sovereign: self-hostable weights only; **document each tool's model licenses** (CC-BY-NC
  is demo-only, never a shipped default).
