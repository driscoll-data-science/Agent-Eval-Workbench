import pytest

from inventory import Inventory


def test_remove_more_than_stock_raises():
    inv = Inventory()
    inv.add("bolt", 2)
    with pytest.raises(ValueError):
        inv.remove("bolt", 3)


def test_stock_unchanged_after_failed_remove():
    inv = Inventory()
    inv.add("bolt", 2)
    with pytest.raises(ValueError):
        inv.remove("bolt", 3)
    assert inv.count("bolt") == 2


def test_remove_exact_stock_ok():
    inv = Inventory()
    inv.add("bolt", 2)
    inv.remove("bolt", 2)
    assert inv.count("bolt") == 0
