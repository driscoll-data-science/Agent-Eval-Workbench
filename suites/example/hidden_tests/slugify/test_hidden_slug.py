from slug import slugify


def test_accents_transliterated():
    assert slugify("Crème Brûlée") == "creme-brulee"


def test_mixed_accents_and_numbers():
    assert slugify("Ñandú 7 días") == "nandu-7-dias"


def test_ascii_unchanged():
    assert slugify("Hello, World!") == "hello-world"


def test_signature_unchanged():
    import inspect

    assert list(inspect.signature(slugify).parameters) == ["text"]
