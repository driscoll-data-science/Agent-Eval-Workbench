from slug import slugify


def test_basic():
    assert slugify("Hello World") == "hello-world"


def test_punctuation():
    assert slugify("Hello, World!") == "hello-world"


def test_numbers():
    assert slugify("Release 2 of 3") == "release-2-of-3"
