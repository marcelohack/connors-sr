"""
Registry for Support & Resistance calculation methods

Provides registration and discovery of external SR calculation methods.
"""

from typing import Any, Callable, Dict, List, Type


class SRMethodRegistry:
    """Registry for SR calculation methods"""

    def __init__(self) -> None:
        self._sr_methods: Dict[str, Any] = {}

    def register_sr_method(self, name: str) -> Callable[[Type], Type]:
        """Register an external Support & Resistance calculation method

        Args:
            name: Unique identifier for the SR method

        Returns:
            Decorator function

        Example:
            @registry.register_sr_method("my_custom_sr")
            class MyCustomSRCalculator(BaseSRCalculator):
                ...
        """

        def decorator(cls: Type) -> Type:
            self._sr_methods[name] = cls
            cls._registry_name = name
            return cls

        return decorator

    def get_sr_method(self, name: str) -> Any:
        """Get an SR method by name

        Args:
            name: The name of the registered SR method

        Returns:
            The registered SR calculator class

        Raises:
            ValueError: If the method name is not registered
        """
        if name not in self._sr_methods:
            raise ValueError(
                f"SR method '{name}' not found. Available: {list(self._sr_methods.keys())}"
            )
        return self._sr_methods[name]

    def list_sr_methods(self) -> List[str]:
        """List available external SR methods

        Returns:
            List of registered SR method names
        """
        return list(self._sr_methods.keys())


# Global registry instance
registry = SRMethodRegistry()
