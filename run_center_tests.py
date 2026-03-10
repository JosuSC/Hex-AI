"""Test large boards with center occupied."""
import time, sys

from board import HexBoard
from solution import SmartPlayer

def make_player(pid):
    sp = SmartPlayer.__new__(SmartPlayer)
    sp.player_id = pid
    return sp

results = {"pass": 0, "fail": 0}

def test_center_taken(size, n_extra=0):
    hb = HexBoard(size)
    center = size // 2
    hb.board[center][center] = 2  # Opponent took center
    # Add some extra pieces near center
    import random
    rng = random.Random(42)
    placed = {(center, center)}
    for _ in range(n_extra):
        while True:
            r = rng.randint(max(0, center-5), min(size-1, center+5))
            c = rng.randint(max(0, center-5), min(size-1, center+5))
            if (r,c) not in placed:
                hb.board[r][c] = rng.choice([1, 2])
                placed.add((r,c))
                break
    sp = make_player(1)
    t0 = time.time()
    move = sp.play(hb)
    elapsed = time.time() - t0
    r, c = move
    valid = 0 <= r < size and 0 <= c < size and hb.board[r][c] == 0
    ok = valid and elapsed <= 5.0
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {size}x{size} center taken, {n_extra} extra: move={move}, time={elapsed:.3f}s")
    if ok: results["pass"] += 1
    else: results["fail"] += 1

print("=" * 70)
print("CENTER-TAKEN LARGE BOARD TESTS")
print("=" * 70)

for size in [200, 500, 1000]:
    test_center_taken(size, 0)
    test_center_taken(size, 10)
    test_center_taken(size, 50)

print(f"\n{'='*70}")
print(f"RESULTS: {results['pass']}/{results['pass']+results['fail']} PASS, {results['fail']} FAIL")
print(f"{'='*70}")
sys.exit(0 if results['fail'] == 0 else 1)
