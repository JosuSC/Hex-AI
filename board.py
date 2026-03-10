"""Minimal HexBoard stub for testing."""

class HexBoard:
    def __init__(self, size=11):
        self.size = size
        self.board = [[0] * size for _ in range(size)]
