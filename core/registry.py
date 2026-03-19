from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Literal

import yaml

from .interfaces import FunctionOrchestratorAdapter, FunctionStrategyAdapter


ComponentKind = Literal["strategy", "orchestrator", "metafilter", "evaluator", "portfolio_allocator"]


def _load_symbol(path: str) -> Any:
    """
    Load a Python object from 'module.submodule:object_name' import path.
    """
    if ":" not in path:
        raise ValueError(f"Invalid import path '{path}'. Expected module.path:object_name")
    module_path, symbol_name = path.split(":", 1)
    module = import_module(module_path)
    if not hasattr(module, symbol_name):
        raise ValueError(f"Symbol '{symbol_name}' not found in module '{module_path}'")
    return getattr(module, symbol_name)


@dataclass
class RegistryEntry:
    name: str
    kind: ComponentKind
    factory: Callable[..., Any]
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def create(self, **kwargs: Any) -> Any:
        merged = dict(self.default_kwargs)
        merged.update(kwargs)
        return self.factory(**merged)


class ComponentRegistry:
    """
    Lightweight config-driven component registry.

    Config format:
    {
      "strategy": {
        "ma_crossover": {
          "path": "strategies.trend:ma_crossover_signals",
          "adapter": "function_strategy",
          "kwargs": {}
        }
      }
    }
    """

    def __init__(self) -> None:
        self._entries: dict[ComponentKind, dict[str, RegistryEntry]] = {
            "strategy": {},
            "orchestrator": {},
            "metafilter": {},
            "evaluator": {},
            "portfolio_allocator": {},
        }

    def register(
        self,
        kind: ComponentKind,
        name: str,
        factory: Callable[..., Any],
        default_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._entries[kind][name] = RegistryEntry(
            name=name,
            kind=kind,
            factory=factory,
            default_kwargs=default_kwargs or {},
        )

    def register_path(
        self,
        kind: ComponentKind,
        name: str,
        import_path: str,
        *,
        adapter: str | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        loaded = _load_symbol(import_path)
        default_kwargs = kwargs or {}

        if adapter == "function_strategy":
            self.register(
                kind=kind,
                name=name,
                factory=lambda **k: FunctionStrategyAdapter(
                    name=name,
                    fn=loaded,
                    default_params=k.get("default_params", {}),
                ),
                default_kwargs={"default_params": default_kwargs},
            )
            return

        if adapter == "function_orchestrator":
            self.register(
                kind=kind,
                name=name,
                factory=lambda **k: FunctionOrchestratorAdapter(
                    name=name,
                    fn=loaded,
                    default_params=k.get("default_params", {}),
                ),
                default_kwargs={"default_params": default_kwargs},
            )
            return

        if adapter in {None, "class"}:
            if callable(loaded):
                self.register(kind=kind, name=name, factory=loaded, default_kwargs=default_kwargs)
                return
            raise ValueError(f"Loaded object for {name} is not callable: {import_path}")

        raise ValueError(f"Unsupported adapter '{adapter}' for component '{name}'.")

    def get_entry(self, kind: ComponentKind, name: str) -> RegistryEntry:
        if name not in self._entries[kind]:
            available = sorted(self._entries[kind].keys())
            raise KeyError(f"{kind} '{name}' not registered. Available: {available}")
        return self._entries[kind][name]

    def create(self, kind: ComponentKind, name: str, **kwargs: Any) -> Any:
        return self.get_entry(kind, name).create(**kwargs)

    def list(self, kind: ComponentKind | None = None) -> dict[str, list[str]] | list[str]:
        if kind is not None:
            return sorted(self._entries[kind].keys())
        return {k: sorted(v.keys()) for k, v in self._entries.items()}

    def load_config(self, config: dict[str, Any]) -> None:
        for raw_kind, components in config.items():
            kind = raw_kind.lower()
            if kind not in self._entries:
                continue
            if not isinstance(components, dict):
                continue
            for name, payload in components.items():
                if not isinstance(payload, dict):
                    raise ValueError(f"Registry payload for {kind}.{name} must be a mapping.")
                import_path = str(payload.get("path", "")).strip()
                if not import_path:
                    raise ValueError(f"Missing 'path' for {kind}.{name}")
                adapter = payload.get("adapter")
                kwargs = payload.get("kwargs", {}) or {}
                self.register_path(
                    kind=kind,  # type: ignore[arg-type]
                    name=name,
                    import_path=import_path,
                    adapter=adapter,
                    kwargs=kwargs,
                )

    def load_yaml(self, path: str | Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Registry config root must be a mapping: {path}")
        self.load_config(data)


def register_default_components(registry: ComponentRegistry) -> ComponentRegistry:
    """Register existing lab components with backward-compatible adapters."""
    registry.register_path(
        "strategy",
        "ma_crossover",
        "strategies.trend:ma_crossover_signals",
        adapter="function_strategy",
    )
    registry.register_path(
        "strategy",
        "rsi_reversal",
        "strategies.mean_reversion:rsi_reversal_signals",
        adapter="function_strategy",
    )
    registry.register_path(
        "strategy",
        "donchian_breakout",
        "strategies.trend:donchian_breakout_signals",
        adapter="function_strategy",
    )
    registry.register_path(
        "strategy",
        "bollinger_fade",
        "strategies.mean_reversion:bollinger_fade_signals",
        adapter="function_strategy",
    )
    registry.register_path(
        "strategy",
        "trend_breakout",
        "strategies.trend_breakout:trend_breakout_signals",
        adapter="function_strategy",
    )
    registry.register_path(
        "strategy",
        "mean_reversion_confirmed",
        "strategies.mean_reversion_confirmed:mean_reversion_confirmed_signals",
        adapter="function_strategy",
    )
    registry.register_path(
        "orchestrator",
        "regime_specialist",
        "orchestrators.regime_specialist:RegimeSpecialistOrchestrator",
        adapter="class",
    )
    registry.register_path(
        "metafilter",
        "rule_based_meta_filter",
        "metalabel.trade_filter:RuleBasedMetaFilter",
        adapter="class",
    )
    registry.register_path(
        "evaluator",
        "walk_forward",
        "research.walk_forward:run_walk_forward",
        adapter="class",
    )
    registry.register_path(
        "portfolio_allocator",
        "v2_portfolio_allocator",
        "portfolio.allocator:V2PortfolioAllocator",
        adapter="class",
    )
    return registry


GLOBAL_REGISTRY = register_default_components(ComponentRegistry())
