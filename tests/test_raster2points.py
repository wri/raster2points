from raster2points.raster2points import _get_steps


class DummyImage(object):
    def __init__(self, block_shapes, width):
        self.block_shapes = block_shapes
        self.width = width


def test__get_steps_strips():
    block_shapes = [(1, 40000)]
    width = 40000
    image = DummyImage(block_shapes, width)

    steps = _get_steps(image, max_size=4096)

    assert steps == (419, 40000)


def test__get_steps_large_strips():
    block_shapes = [(1, 40000)]
    width = 40000
    image = DummyImage(block_shapes, width)

    steps = _get_steps(image, max_size=256)

    assert steps == (1, 40000)


def test__get_steps_blocks():
    block_shapes = [(256, 256)]
    width = 40000
    image = DummyImage(block_shapes, width)

    steps = _get_steps(image, max_size=4096)

    assert steps == (4096, 4096)


def test__get_steps_large_blocks():
    block_shapes = [(256, 256)]
    width = 40000
    image = DummyImage(block_shapes, width)

    steps = _get_steps(image, max_size=128)

    assert steps == (256, 256)
