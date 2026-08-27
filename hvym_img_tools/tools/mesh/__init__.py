"""mesh — sketch to an untextured 3D reference (tool #2)."""
from ...core.registry import register
from .tool import MeshInput, MeshTool

register(MeshTool)

__all__ = ["MeshTool", "MeshInput"]
