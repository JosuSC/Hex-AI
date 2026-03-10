"""Extended stress tests: larger games, 13x13 mid-game, large board with pieces."""
import time, sys

from board import HexBoard
from player import Player
from solution import (SmartPlayer, build_neighbor_table, check_connection,
                      dijkstra_distance, build_dsu_from_board, dsu_check_win,
                      dsu_add_stone)

def make_hex_board(size, pieces=None):
    hb = HexBoard(size)
    if pieces:
        for r, c, p in pieces:
            hb.board[r][c] = p
    return hb

def make_player(pid):
    sp = SmartPlayer.__new__(SmartPlayer)
    sp.player_id = pid
    return sp

results = {"pass": 0, "fail": 0}

def record(name, passed):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    if passed:
        results["pass"] += 1
    else:
        results["fail"] += 1

def test_full_game(size, max_turns=None):
    if max_turns is None:
        max_turns = size * size
    board = [[0]*size for _ in range(size)]
    ntable = build_neighbor_table(size)
    p1, p2 = make_player(1), make_player(2)
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
        if board[r][c] != 0:
            record(f"full_game_{size}: INVALID MOVE {move} turn {turn}", False)
            return
        board[r][c] = current
        if check_connection(board, size, current, ntable):
            winner = current
            break
        current = 3 - current
        turn += 1
    max_t = max(times) if times else 0
    avg_t = sum(times)/len(times) if times else 0
    timeout_v = sum(1 for t in times if t > 5.0)
    ok = winner is not None and timeout_v == 0
    record(f"full_game_{size}x{size}: winner=P{winner}, turns={turn+1}, max={max_t:.3f}s, avg={avg_t:.3f}s, timeouts={timeout_v}", ok)

def test_midgame_13x13():
    import random as _rng
    for pct in [0.1, 0.3, 0.5]:
        r = _rng.Random(42)
        size = 13
        board = [[0]*size for _ in range(size)]
        cells = [(i,j) for i in range(size) for j in range(size)]
        r.shuffle(cells)
        n = int(size*size*pct)
        p1c = p2c = 0
        for idx in range(n):
            ri, ci = cells[idx]
            if p1c <= p2c:
                board[ri][ci] = 1; p1c += 1
            else:
                board[ri][ci] = 2; p2c += 1
        player = 1 if p1c <= p2c else 2
        hb = make_hex_board(size)
        hb.board = board
        sp = make_player(player)
        t0 = time.time()
        move = sp.play(hb)
        elapsed = time.time() - t0
        valid = (0 <= move[0] < size and 0 <= move[1] < size and board[move[0]][move[1]] == 0)
        ok = valid and elapsed <= 5.0
        record(f"midgame_13x13_{int(pct*100)}%: move={move}, time={elapsed:.3f}s", ok)

def test_large_board_with_pieces(size):
    """Large board with a few pieces placed — more realistic than empty."""
    import random as _rng
    r = _rng.Random(42)
    board = [[0]*size for _ in range(size)]
    # Place 20 random pieces
    cells = [(i,j) for i in range(size) for j in range(size)]
    r.shuffle(cells)
    for idx in range(20):
        ri, ci = cells[idx]
        board[ri][ci] = 1 if idx % 2 == 0 else 2
    hb = make_hex_board(size)
    hb.board = board
    sp = make_player(1)
    t0 = time.time()
    move = sp.play(hb)
    elapsed = time.time() - t0
    valid = (0 <= move[0] < size and 0 <= move[1] < size and board[move[0]][move[1]] == 0)
    ok = valid and elapsed <= 5.0
    record(f"large_board_{size}x{size}_with_pieces: move={move}, time={elapsed:.3f}s", ok)

def test_immediate_win_edge_cases():
    """Win-in-1 where the winning cell is in the corner."""
    # P1 needs (0,0) to connect col 0 and finish
    size = 7
    pieces = [(0, c, 1) for c in range(1, 7)]  # row 0, cols 1..6
    pieces += [(1, 0, 2), (2, 0, 2)]
    hb = make_hex_board(size, pieces)
    sp = make_player(1)
    t0 = time.time()
    move = sp.play(hb)
    elapsed = time.time() - t0
    # (0,0) wins: row 0 from col 0 to col 6
    ntable = build_neighbor_table(size)
    nb = [row[:] for row in hb.board]
    nb[move[0]][move[1]] = 1
    won = check_connection(nb, size, 1, ntable)
    record(f"win_corner_7x7: move={move}, wins={won}, time={elapsed:.3f}s", won and elapsed <= 5.0)

print("=" * 70)
print("EXTENDED STRESS TESTS")
print("=" * 70)

print("\n--- Full Game 7x7 ---")
test_full_game(7)

print("\n--- 13x13 Mid-Game ---")
test_midgame_13x13()

print("\n--- Large Boards with Pieces ---")
for s in [200, 500]:
    test_large_board_with_pieces(s)

print("\n--- Win-in-1 Corner ---")
test_immediate_win_edge_cases()

print(f"\n{'='*70}")
print(f"EXTENDED: {results['pass']}/{results['pass']+results['fail']} PASS, {results['fail']} FAIL")
print(f"{'='*70}")
sys.exit(0 if results['fail'] == 0 else 1)
