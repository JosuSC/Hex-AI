"""
Exhaustive test suite for HEX engine (solution.py).
Tests all scenarios from TEST_REPORT.md and beyond.
"""

import time
import sys
import traceback

# We need to create a mock HexBoard and Player if they don't exist properly
# First try to import directly
try:
    from board import HexBoard
    from player import Player
except ImportError:
    # Create minimal stubs
    class Player:
        pass

    class HexBoard:
        def __init__(self, size, board=None):
            self.size = size
            if board is not None:
                self.board = board
            else:
                self.board = [[0]*size for _ in range(size)]

from solution import SmartPlayer, build_neighbor_table, dijkstra_distance, evaluate_fast, evaluate_board, check_connection, build_dsu_from_board, dsu_check_win, dsu_add_stone

# ============================================================
# Helpers
# ============================================================

def make_board(size, pieces=None):
    """Create a board with given pieces. pieces = [(r, c, player), ...]"""
    b = [[0]*size for _ in range(size)]
    if pieces:
        for r, c, p in pieces:
            b[r][c] = p
    return b

def make_hex_board(size, pieces=None):
    b = make_board(size, pieces)
    hb = HexBoard(size)
    hb.board = b
    hb.size = size
    return hb

def make_player(player_id):
    sp = SmartPlayer.__new__(SmartPlayer)
    sp.player_id = player_id
    return sp

def timed_play(sp, hb, label="", time_limit=5.0):
    """Plays and measures time. Returns (move, elapsed, passed)."""
    t0 = time.time()
    move = sp.play(hb)
    elapsed = time.time() - t0
    passed = elapsed <= time_limit
    r, c = move
    valid = (0 <= r < hb.size and 0 <= c < hb.size and hb.board[r][c] == 0)
    status = "PASS" if (passed and valid) else "FAIL"
    details = []
    if not valid:
        details.append("INVALID MOVE")
    if not passed:
        details.append(f"TIMEOUT {elapsed:.3f}s")
    detail_str = " ".join(details) if details else f"{elapsed:.3f}s"
    print(f"  [{status}] {label}: move={move} {detail_str}")
    return move, elapsed, passed and valid

# ============================================================
# Test categories
# ============================================================

results = {"pass": 0, "fail": 0, "tests": []}

def record(name, passed):
    results["tests"].append((name, passed))
    if passed:
        results["pass"] += 1
    else:
        results["fail"] += 1

# ------ 1. IMMEDIATE WIN DETECTION (Bug 1 & 2) ------

def test_immediate_win_p1_3x3():
    """P1 can win in 1 move on 3x3 by placing at (1,0)."""
    pieces = [(0, 0, 1), (1, 1, 1), (1, 2, 1),  # P1 chain missing (1,0)
              (0, 1, 2), (0, 2, 2), (2, 0, 2)]
    hb = make_hex_board(3, pieces)
    sp = make_player(1)
    move, elapsed, ok = timed_play(sp, hb, "P1 win-in-1 3x3")
    # Check that the move wins
    ntable = build_neighbor_table(3)
    new_board = [row[:] for row in hb.board]
    new_board[move[0]][move[1]] = 1
    won = check_connection(new_board, 3, 1, ntable)
    passed = ok and won
    record("immediate_win_p1_3x3", passed)
    if not won:
        print(f"    WARNING: Move {move} does not win! Board needs (1,0)")

def test_immediate_win_p1_11x11():
    """P1 has a chain from col 0 to col 9, missing only (5,10). Must detect."""
    size = 11
    pieces = [(5, c, 1) for c in range(10)]  # P1: row 5, cols 0..9
    # Add some P2 pieces to not interfere
    pieces += [(0, c, 2) for c in range(5)]
    hb = make_hex_board(size, pieces)
    sp = make_player(1)
    move, elapsed, ok = timed_play(sp, hb, "P1 win-in-1 11x11")
    passed = ok and move == (5, 10)
    record("immediate_win_p1_11x11", passed)
    if move != (5, 10):
        print(f"    BUG: Expected (5,10) but got {move}")

def test_immediate_win_p2_11x11():
    """P2 has chain from row 0 to row 9 (connected), missing 1 cell to reach row 10."""
    size = 11
    # Build a P2 chain connected top-to-bottom following hex adjacency
    # In Even-R offset, row r connects to row r+1 at specific columns
    # Simple approach: P2 occupies col 5 rows 0..9 — find which (10,x) wins
    pieces = [(r, 5, 2) for r in range(10)]
    pieces += [(3, 0, 1), (3, 1, 1), (4, 0, 1)]  # some P1 pieces
    hb = make_hex_board(size, pieces)
    sp = make_player(2)
    move, elapsed, ok = timed_play(sp, hb, "P2 win-in-1 11x11")
    # Verify the move actually wins
    ntable = build_neighbor_table(size)
    new_board = [row[:] for row in hb.board]
    new_board[move[0]][move[1]] = 2
    won = check_connection(new_board, size, 2, ntable)
    passed = ok and won
    record("immediate_win_p2_11x11", passed)
    if not won:
        print(f"    BUG: Move {move} does not win for P2")

def test_immediate_win_p1_15x15():
    """P1 has chain from col 0 to col 13, missing (7,14)."""
    size = 15
    pieces = [(7, c, 1) for c in range(14)]
    pieces += [(0, c, 2) for c in range(5)]
    hb = make_hex_board(size, pieces)
    sp = make_player(1)
    move, elapsed, ok = timed_play(sp, hb, "P1 win-in-1 15x15")
    passed = ok and move == (7, 14)
    record("immediate_win_p1_15x15", passed)
    if move != (7, 14):
        print(f"    BUG: Expected (7,14) but got {move}")

# ------ 2. BLOCKING DETECTION (Bug 3) ------

def test_blocking_5x5():
    """
    P1 is near winning on row 2. P2 should block a critical cell.
    The ONLY winning cell for P1 is (2,4). P2 MUST play there.
    """
    size = 5
    # P1 has cells on row 2, cols 0..3 (needs (2,4) to win)
    pieces = [(2, 0, 1), (2, 1, 1), (2, 2, 1), (2, 3, 1)]
    # P2 has some pieces  
    pieces += [(0, 0, 2), (0, 1, 2), (1, 0, 2)]
    hb = make_hex_board(size, pieces)
    sp = make_player(2)
    move, elapsed, ok = timed_play(sp, hb, "P2 block 5x5")
    # P2 should block at (2,4) — the only cell that completes P1's row
    blocked_critical = (move == (2, 4))
    record("blocking_5x5", ok and blocked_critical)
    if not blocked_critical:
        print(f"    WARNING: Expected (2,4) to block P1 win, got {move}")

# ------ 3. OPENING TESTS (various sizes) ------

def test_opening(size, label):
    """Empty board opening should return center."""
    hb = make_hex_board(size)
    sp = make_player(1)
    move, elapsed, ok = timed_play(sp, hb, f"Opening {label}")
    center = size // 2
    is_center = move == (center, center)
    record(f"opening_{label}", ok and is_center)
    if not is_center:
        print(f"    NOTE: Expected ({center},{center}) but got {move}")

# ------ 4. LARGE BOARD TIMEOUT (Bug timeout) ------

def test_large_board_opening(size, label):
    """Large board opening must complete within 5 seconds."""
    hb = make_hex_board(size)
    sp = make_player(1)
    move, elapsed, ok = timed_play(sp, hb, f"Large opening {label}")
    center = size // 2
    is_center = move == (center, center)
    record(f"large_opening_{label}", ok)  # Just need valid + within time
    return elapsed

# ------ 5. MID-GAME TESTS ------

def test_midgame(size, fill_pct, seed=42):
    """Mid-game with random fill."""
    import random as _rng
    r = _rng.Random(seed)
    board = [[0]*size for _ in range(size)]
    cells = [(i, j) for i in range(size) for j in range(size)]
    r.shuffle(cells)
    n_fill = int(size * size * fill_pct)
    p1_count = 0
    p2_count = 0
    for idx in range(n_fill):
        ri, ci = cells[idx]
        if p1_count <= p2_count:
            board[ri][ci] = 1
            p1_count += 1
        else:
            board[ri][ci] = 2
            p2_count += 1
    # Next to play is P1 if counts equal, else P2
    player = 1 if p1_count <= p2_count else 2
    hb = make_hex_board(size, None)
    hb.board = board
    sp = make_player(player)
    label = f"Midgame {size}x{size} {int(fill_pct*100)}%"
    move, elapsed, ok = timed_play(sp, hb, label)
    record(f"midgame_{size}_{int(fill_pct*100)}", ok)

# ------ 6. EDGE CASES ------

def test_1x1():
    hb = make_hex_board(1)
    sp = make_player(1)
    move, elapsed, ok = timed_play(sp, hb, "1x1 board")
    passed = ok and move == (0, 0)
    record("1x1", passed)

def test_2x2():
    hb = make_hex_board(2)
    sp = make_player(1)
    move, elapsed, ok = timed_play(sp, hb, "2x2 board")
    record("2x2", ok)

def test_single_empty():
    """Only 1 empty cell left."""
    size = 5
    board = [[1 if (r+c) % 2 == 0 else 2 for c in range(size)] for r in range(size)]
    board[0][0] = 0  # Only empty cell
    hb = make_hex_board(size)
    hb.board = board
    sp = make_player(1)
    move, elapsed, ok = timed_play(sp, hb, "Single empty cell")
    passed = ok and move == (0, 0)
    record("single_empty", passed)

def test_determinism():
    """Same input -> same output across 5 calls."""
    hb = make_hex_board(5)
    sp = make_player(1)
    moves = set()
    for i in range(5):
        m = sp.play(hb)
        moves.add(m)
    passed = len(moves) == 1
    print(f"  [{'PASS' if passed else 'FAIL'}] Determinism 5x5: {len(moves)} distinct moves: {moves}")
    record("determinism", passed)

def test_immutability():
    """play() must not modify the original board."""
    hb = make_hex_board(7)
    original = [row[:] for row in hb.board]
    sp = make_player(1)
    sp.play(hb)
    same = all(hb.board[r][c] == original[r][c] for r in range(7) for c in range(7))
    print(f"  [{'PASS' if same else 'FAIL'}] Immutability: board unchanged={same}")
    record("immutability", same)

# ------ 7. FULL GAME SIMULATION ------

def test_full_game(size, max_turns=None):
    """Simulate a full game between two SmartPlayers."""
    if max_turns is None:
        max_turns = size * size
    board = [[0]*size for _ in range(size)]
    ntable = build_neighbor_table(size)
    p1 = make_player(1)
    p2 = make_player(2)
    current = 1
    times = []
    turn = 0
    winner = None

    while turn < max_turns:
        hb = make_hex_board(size)
        hb.board = [row[:] for row in board]
        sp = p1 if current == 1 else p2
        t0 = time.time()
        move = sp.play(hb)
        elapsed = time.time() - t0
        times.append(elapsed)

        r, c = move
        if board[r][c] != 0 or r < 0 or r >= size or c < 0 or c >= size:
            print(f"    INVALID MOVE by P{current}: {move} on turn {turn}")
            record(f"full_game_{size}", False)
            return

        board[r][c] = current
        if check_connection(board, size, current, ntable):
            winner = current
            break

        current = 3 - current
        turn += 1

    max_t = max(times) if times else 0
    avg_t = sum(times) / len(times) if times else 0
    timeout_violations = sum(1 for t in times if t > 5.0)
    ok = winner is not None and timeout_violations == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] Full game {size}x{size}: winner=P{winner}, "
          f"turns={turn+1}, max_t={max_t:.3f}s, avg_t={avg_t:.3f}s, timeouts={timeout_violations}")
    record(f"full_game_{size}", ok)

# ============================================================
# Main execution
# ============================================================

def run_all_tests():
    print("=" * 70)
    print("HEX ENGINE EXHAUSTIVE TEST SUITE")
    print("=" * 70)

    # Phase 1: Immediate win detection (Bugs 1 & 2)
    print("\n--- Phase 1: Immediate Win Detection ---")
    test_immediate_win_p1_3x3()
    test_immediate_win_p1_11x11()
    test_immediate_win_p2_11x11()
    test_immediate_win_p1_15x15()

    # Phase 2: Blocking detection (Bug 3)
    print("\n--- Phase 2: Blocking Detection ---")
    test_blocking_5x5()

    # Phase 3: Opening tests
    print("\n--- Phase 3: Opening Tests ---")
    for s in [3, 5, 7, 11, 13, 15]:
        test_opening(s, f"{s}x{s}")

    # Phase 4: Large board timeout
    print("\n--- Phase 4: Large Board Scalability ---")
    for s in [20, 50, 100, 200, 500, 1000]:
        test_large_board_opening(s, f"{s}x{s}")

    # Phase 5: Mid-game tests
    print("\n--- Phase 5: Mid-Game Tests ---")
    for s in [5, 7, 11]:
        for pct in [0.1, 0.3, 0.5]:
            test_midgame(s, pct)

    # Phase 6: Edge cases
    print("\n--- Phase 6: Edge Cases ---")
    test_1x1()
    test_2x2()
    test_single_empty()
    test_determinism()
    test_immutability()

    # Phase 7: Full game simulations
    print("\n--- Phase 7: Full Game Simulations ---")
    test_full_game(3)
    test_full_game(5)

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {results['pass']}/{results['pass']+results['fail']} PASS, {results['fail']} FAIL")
    print("=" * 70)

    failed = [(name, p) for name, p in results["tests"] if not p]
    if failed:
        print("\nFailed tests:")
        for name, _ in failed:
            print(f"  - {name}")

    return results["fail"] == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
