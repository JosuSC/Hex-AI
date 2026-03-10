"""Test large boards with many pieces to verify <5s constraint."""
import time, sys, random

from board import HexBoard
from player import Player
from solution import SmartPlayer

def make_player(pid):
    sp = SmartPlayer.__new__(SmartPlayer)
    sp.player_id = pid
    return sp

results = {"pass": 0, "fail": 0}

def test_large_with_pieces(size, n_pieces, seed=42):
    rng = random.Random(seed)
    hb = HexBoard(size)
    cells = [(r, c) for r in range(size) for c in range(size)]
    rng.shuffle(cells)
    p1c = p2c = 0
    for i in range(n_pieces):
        r, c = cells[i]
        if p1c <= p2c:
            hb.board[r][c] = 1; p1c += 1
        else:
            hb.board[r][c] = 2; p2c += 1
    player = 1 if p1c <= p2c else 2
    sp = make_player(player)
    t0 = time.time()
    move = sp.play(hb)
    elapsed = time.time() - t0
    r, c = move
    valid = 0 <= r < size and 0 <= c < size and hb.board[r][c] == 0
    ok = valid and elapsed <= 5.0
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {size}x{size} with {n_pieces} pieces: move={move}, time={elapsed:.3f}s")
    if ok: results["pass"] += 1
    else: results["fail"] += 1

print("=" * 70)
print("LARGE BOARD PERFORMANCE TESTS")
print("=" * 70)

# Various sizes and piece counts
for size in [200, 500, 1000]:
    for n_pieces in [2, 10, 50, 100]:
        if n_pieces < size * size:
            test_large_with_pieces(size, n_pieces)

print(f"\n{'='*70}")
print(f"RESULTS: {results['pass']}/{results['pass']+results['fail']} PASS, {results['fail']} FAIL")
print(f"{'='*70}")
sys.exit(0 if results['fail'] == 0 else 1)
