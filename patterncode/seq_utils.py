import itertools

import numba
from numba import njit
import numpy as np

from patterncode.config import IUPAC_DNA


def pad_pattern(pattern: str, target_len: int):
    """
    Pad pattern with Ns to target length

    :param pattern: pattern to pad
    :param target_len: target length
    :return: padded pattern
    """
    k = len(pattern)
    n = target_len

    assert k <= n

    if k < n:
        padded = pattern + 'N' * (n - k)
    else:
        padded = pattern
    return padded


def expand_iupac(sequence: str) -> list[str]:
    """
    Expand IUPAC codes in sequence to all possible bases

    :param sequence:
    :return: list of sequences matching the IUPAC coded sequence
    """
    # Convert input sequence to list of possible bases
    sequence_options = [IUPAC_DNA[base] for base in sequence]

    # Generate all possible combinations
    possible_sequences = list(itertools.product(*sequence_options))

    # Convert tuples to strings
    expanded_sequences = [''.join(sequence) for sequence in possible_sequences]

    return expanded_sequences


def all_patterns(n: int, k: int = 4) -> list[str]:
    # codes = ['A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', 'N']
    # codes = ['A', 'C', 'G', 'T', 'R', 'Y', 'S', 'W', 'K', 'M', 'N']
    codes = {'A', 'C', 'G', 'T', 'N'}
    determined = {'A', 'C', 'G', 'T'}

    sequences = [""]
    for _ in range(n):
        new_sequences = []
        for seq in sequences:
            for letter in codes:
                new_sequences.append(seq + letter)
        sequences = new_sequences

    valid_sequences = [seq for seq in sequences if sum(
        1 for letter in seq if letter in determined) >= k]
    return valid_sequences


def test_all_patterns():
    def is_valid(sequence: str, k: int) -> bool:
        set_b = {'A', 'C', 'G', 'T'}
        count = sum(1 for letter in sequence if letter in set_b)
        return count >= k

    test_cases = [
        (3, 2),
        (4, 3),
        (3, 3),
        (2, 1),
        (5, 4),
    ]

    for n, k in test_cases:
        sequences = all_patterns(n, k)
        print(
            f"Testing with n={n}, k={k}: {len(sequences)} sequences generated.")

        for seq in sequences:
            assert len(
                seq) == n, f"Invalid length: expected {n}, got {len(seq)}"
            assert is_valid(
                seq, k), f"Invalid sequence: {seq}, expected at least {k} letters from set_b"

        print("All sequences are valid.")


def test_expand_iupac():
    test_cases = [
        ('CTTAG', ['CTTAG']),
        ('ATR', ['ATA', 'ATG']),
        ('YCG', ['CCG', 'TCG']),
        ('GS', ['GG', 'GC']),
        ('TCNG', ['TCAG', 'TCCG', 'TCGG', 'TCTG']),
        ('.-', ['.-']),
    ]

    for sequence, expected_output in test_cases:
        result = expand_iupac(sequence)
        assert set(result) == set(
            expected_output), f"Error for sequence {sequence}: expected {expected_output}, got {result}"


@njit
def check_match(string, substring, pos):
    for i, char in enumerate(substring):
        if string[pos + i] not in char:
            return False
    return True


@njit
def find_pattern_positions(sequence: str, pattern: str):
    """
    Find all positions of pattern in sequence

    :param sequence: genome sequence over {A, C, G, T}
    :param pattern: sequence to find over {A, C, G, T, R, Y, S, W, K, M, B, D, H, V, N} (IUPAC codes)
    :return: list of positions where the pattern is found
    """
    expanded_pattern = [iupac_to_bases(char) for char in pattern]
    positions = []
    for i in range(len(sequence) - len(pattern) + 1):
        if check_match(sequence, expanded_pattern, i):
            positions.append(i)
    return np.array(positions)


@njit(nogil=True, boundscheck=False, cache=False)
def pack_string(input_string: bytes, window_size: int):
    """
    pack string using two bits per letter into 16-bit integers,
    each rolling window of size window_size is packed into one integer

    :param input_string: the string to pack
    :param window_size: the size of the window to pack
    :return: packed string into 16-bit integers
    """
    mask = numba.int16((1 << (window_size * 2)) - 1)
    assert 1 <= window_size <= 8

    n = len(input_string)
    packed_ints = np.zeros(n - window_size + 1, dtype=np.int16)
    rolling_val = numba.int16(0)

    for i, c in enumerate(input_string):
        ec = None
        if c == ord('A'):
            ec = 0b00
        elif c == ord('C'):
            ec = 0b01
        elif c == ord('G'):
            ec = 0b10
        elif c == ord('T'):
            ec = 0b11

        if ec is None:
            rolling_val = -1
        elif rolling_val != -1:
            rolling_val = ((rolling_val << 2) | ec) & mask
        else:
            rolling_val = ec

        j = i - window_size + 1
        if j >= 0:
            packed_ints[j] = rolling_val

    return packed_ints


@njit
def iupac_to_bases(iupac_code):
    iupac_dict = {
        "R": "AG",
        "Y": "CT",
        "S": "GC",
        "W": "AT",
        "K": "GT",
        "M": "AC",
        "B": "CGT",
        "D": "AGT",
        "H": "ACT",
        "V": "ACG",
        "N": "ACGT"
    }
    return iupac_dict.get(iupac_code, iupac_code)


def test_find_all_substring_positions():
    assert np.array_equal(find_pattern_positions("ATGCGGTGGTAGCTACG", "RTG"), [0, 5])
