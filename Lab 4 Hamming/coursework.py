"""
This file is part of Lab 4 (Hamming Codes), assessed coursework for the module
COMP70103 Statistical Information Theory. 

You should submit an updated version of this file, replacing the
NotImplementedError's with the correct implementation of each function. Do not
edit any other functions.

Follow further instructions in the attached .pdf and .ipynb files, available
through Scientia.
"""
from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import numpy.random as rn
from itertools import product

alphabet = "abcdefghijklmnopqrstuvwxyz01234567890 .,\n"
digits = "0123456789"

def char2bits(char: chr) -> np.array:
    '''
    Given a character in the alphabet, returns a 8-bit numpy array of 0,1 which represents it
    '''
    num   = ord(char)
    if num >= 256:
        raise ValueError("Character not recognised.")
    bits = format(num, '#010b')
    bits  = [ int(b) for b in bits[2:] ]
    return np.array(bits)


def bits2char(bits) -> chr:
    '''
    Given a 7-bit numpy array of 0,1 or bool, returns the character it encodes
    '''
    bits  = ''.join(bits.astype(int).astype(str))
    num   =  int(bits, base = 2)
    return chr(num)


def text2bits(text: str) -> np.ndarray:
    '''
    Given a string, returns a numpy array of bool with the binary encoding of the text
    '''
    text = text.lower()
    text = [ t for t in text if t in alphabet ]
    bits = [ char2bits(c) for c in text ]
    return np.array(bits, dtype = bool).flatten()


def bits2text(bits: np.ndarray) -> str:
    '''
    Given a numpy array of bool or 0 and 1 (as int) which represents the
    binary encoding a text, return the text as a string
    '''
    if np.mod(len(bits), 8) != 0:
        raise ValueError("The length of the bit string must be a multiple of 8.")
    bits = bits.reshape(int(len(bits)/8), 8)
    chrs = [ bits2char(b) for b in bits ]
    return ''.join(chrs)


def parity_matrix(m : int) -> np.ndarray:
    """
    m : int
      The number of parity bits to use

    return : np.ndarray
      m-by-n parity check matrix
    """
    n = 2 ** m - 1 
    H = []

    for index in range(1, n + 1):
        binary_string = format(index, "b").zfill(m)
        binary_list = list( map(int, binary_string) )
        H.append( binary_list ) 

    return np.array(H).T[::-1]

def hamming_generator(m : int) -> np.ndarray:
    """
    m : int
      The number of parity bits to use

    return : np.ndarray
      k-by-n generator matrix
    """
    def is_data_bit(index: int) -> bool: 
        return (index & (index - 1)) == 0
    
    n = 2 ** m - 1 
    k = n - m 

    H = parity_matrix(m)
    H_T = H.T

    data_indices =  [ 
        index - 1 for index in range(1, n + 1) 
        if not is_data_bit(index)
    ]
    
    data_bits = H_T[data_indices].T

    G_T = []
    identity_matrix = np.eye(k, dtype=int)
    data_idex, identity_idex = 0, 0

    for index in range(1, n + 1): 
        G_T.append(
            data_bits[data_idex] if is_data_bit(index)
            else identity_matrix[identity_idex]
        )
        data_idex += is_data_bit(index)
        identity_idex += not is_data_bit(index)

    return np.array(G_T).T


def hamming_encode(data : np.ndarray, m : int) -> np.ndarray:
    """
    data : np.ndarray
      array of shape (k,) with the block of bits to encode

    m : int
      The number of parity bits to use

    return : np.ndarray
      array of shape (n,) with the corresponding Hamming codeword
    """
    assert( data.shape[0] == 2 ** m - m - 1 )
    
    G = hamming_generator(m) 
    codeword = np.dot(G.T, data) % 2 
    
    return np.array(codeword).astype(int)


def hamming_decode(code : np.ndarray, m : int) -> np.ndarray:
    """
    code : np.ndarray
      Array of shape (n,) containing a Hamming codeword computed with m parity bits
    m : int
      Number of parity bits used when encoding

    return : np.ndarray
      Array of shape (k,) with the decoded and corrected data
    """
    assert(np.log2(len(code) + 1) == int(np.log2(len(code) + 1)) == m)

    H = parity_matrix(m)
    k = 2 ** m - m - 1
    
    z = np.dot(H, code) % 2

    if not all(z[index] == 0 for index in range(0, m)):
        error_position = int("".join(map(str, z[::-1])), 2) - 1
        code[error_position] ^= 1
        print("error position: ", error_position)

    data_indices =  [ 
        index - 1 for index in range(1, m + k + 1) 
        if not (index & (index - 1)) == 0
    ]

    return code[data_indices]


def decode_secret(msg : np.ndarray) -> str:
    """
    msg : np.ndarray
      One-dimensional array of binary integers

    return : str
      String with decoded text
    """
    m = 4  # <-- Your guess goes here
    n = 2 ** m - 1 
    
    if len(msg) % n != 0:
        raise ValueError("Message length is not a multiple of codeword length.")
    
    codeword_chunks = np.split(msg, len(msg) // n)
    
    decoded_bits = []
    for codeword in codeword_chunks:
        decoded_chunk = hamming_decode(codeword, m)
        decoded_bits.extend(decoded_chunk)

    text = bits2text(np.array(decoded_bits))

    if len(text.strip()) > 0: 
        print(f"Decoded successfully with m = {m} \n")
        return text

    return ValueError(f"Decode was unsuccessful with m = {m}")   


def binary_symmetric_channel(data : np.ndarray, p : float) -> np.ndarray:
    """
    data : np.ndarray
      1-dimensional array containing a stream of bits
    p : float
      probability by which each bit is flipped

    return : np.ndarray
      data with a number of bits flipped
    """

    raise NotImplementedError


def decoder_accuracy(m : int, p : float) -> float:
    """
    m : int
      The number of parity bits in the Hamming code being tested
    p : float
      The probability of each bit being flipped

    return : float
      The probability of messages being correctly decoded with this
      Hamming code, using the noisy channel of probability p
    """

    raise NotImplementedError

