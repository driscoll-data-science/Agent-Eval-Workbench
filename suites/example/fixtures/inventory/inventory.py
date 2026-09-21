"""Minimal inventory model used by the example suite."""


class Inventory:
    def __init__(self) -> None:
        self._stock: dict[str, int] = {}

    def add(self, name: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("qty must be positive")
        self._stock[name] = self._stock.get(name, 0) + qty

    def remove(self, name: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("qty must be positive")
        self._stock[name] = self._stock.get(name, 0) - qty

    def count(self, name: str) -> int:
        return self._stock.get(name, 0)
