"""SR method registry — delegates to the unified ComponentRegistry in connors-core.

All SR method registrations are stored in connors-core's central storage backend.
"""

from connors_core.core.registry import ComponentRegistry, registry

# Backward-compatible alias: ``SRMethodRegistry()`` creates a ComponentRegistry
# that has register_sr_method, get_sr_method, list_sr_methods, etc.
SRMethodRegistry = ComponentRegistry

__all__ = ["SRMethodRegistry", "registry"]
