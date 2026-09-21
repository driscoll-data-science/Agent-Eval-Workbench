import pytest

from inventory import Inventory


def test_add_and_count():
    inv = Inventory()
    inv.add("bolt", 5)
    assert inv.count("bolt") == 5


def test_remove_reduces():
    inv = Inventory()
    inv.add("bolt", 5)
    inv.remove("bolt", 2)
    assert inv.count("bolt") == 3


def test_rejects_nonpositive():
    inv = Inventory()
    with pytest.raises(ValueError):
        inv.add("bolt", 0)
