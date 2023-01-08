class Keys(object):
    """
    An enum identifying keys on the keyboard
    """
    NONE = None  # Code for "no key press"
    RETURN = 'RETURN'
    SPACE = 'SPACE'
    DELETE = 'DELETE'
    LSHIFT = 'LSHIFT'
    RSHIFT = 'RSHIFT'
    TAB = 'TAB'
    ESC = "ESC"
    RIGHT = 'RIGHT'
    LEFT = 'LEFT'
    UP = 'UP'
    DOWN = 'DOWN'
    DASH = 'DASH'
    EQUALS = 'EQUALS'
    BACKSPACE = 'BACKSPACE'
    LBRACE = 'LBRACE'
    RBRACE = 'RBRACE'
    BACKSLASH = 'BACKSLASH'
    SEMICOLON = 'SEMICOLON'
    APOSTROPHE = 'APOSTROPHE'
    COMMA = 'COMMA'
    PERIOD = 'PERIOD'
    SLASH = 'SLASH'
    A = 'A'
    B = 'B'
    C = 'C'
    D = 'D'
    E = 'E'
    F = 'F'
    G = 'G'
    H = 'H'
    I = 'I'
    J = 'J'
    K = 'K'
    L = 'L'
    M = 'M'
    N = 'N'
    O = 'O'
    P = 'P'
    Q = 'Q'
    R = 'R'
    S = 'S'
    T = 'T'
    U = 'U'
    V = 'V'
    W = 'W'
    X = 'X'
    Y = 'Y'
    Z = 'Z'
    n0 = '0'
    n1 = '1'
    n2 = '2'
    n3 = '3'
    n4 = '4'
    n5 = '5'
    n6 = '6'
    n7 = '7'
    n8 = '8'
    n9 = '9'
    np0 = '0'
    np1 = '1'
    np2 = '2'
    np3 = '3'
    np4 = '4'
    np5 = '5'
    np6 = '6'
    np7 = '7'
    np8 = '8'
    np9 = '9'
    UNKNOWN = 'UNKNOWN'


_keydict = {
    # On a MAC these are the key codes
    -1: Keys.NONE,  # -1 indicates "no key press"
    27: Keys.ESC,
    13: Keys.RETURN,
    32: Keys.SPACE,
    255: Keys.DELETE,
    225: Keys.LSHIFT,
    226: Keys.RSHIFT,
    9: Keys.TAB,
    81: Keys.LEFT,
    82: Keys.UP,
    83: Keys.RIGHT,
    84: Keys.DOWN,
    45: Keys.DASH,
    61: Keys.EQUALS,
    8: Keys.BACKSPACE,
    91: Keys.LBRACE,
    93: Keys.RBRACE,
    92: Keys.BACKSLASH,
    59: Keys.SEMICOLON,
    39: Keys.APOSTROPHE,
    44: Keys.COMMA,
    46: Keys.PERIOD,
    47: Keys.SLASH,
    63: Keys.SLASH,  # On on thinkpad at least
    97: Keys.A,
    98: Keys.B,
    99: Keys.C,
    100: Keys.D,
    101: Keys.E,
    102: Keys.F,
    103: Keys.G,
    104: Keys.H,
    105: Keys.I,
    106: Keys.J,
    107: Keys.K,
    108: Keys.L,
    109: Keys.M,
    110: Keys.N,
    111: Keys.O,
    112: Keys.P,
    113: Keys.Q,
    114: Keys.R,
    115: Keys.S,
    116: Keys.T,
    117: Keys.U,
    118: Keys.V,
    119: Keys.W,
    120: Keys.X,
    121: Keys.Y,
    122: Keys.Z,
    48: Keys.n0,
    49: Keys.n1,
    50: Keys.n2,
    51: Keys.n3,
    52: Keys.n4,
    53: Keys.n5,
    54: Keys.n6,
    55: Keys.n7,
    56: Keys.n8,
    57: Keys.n9,
    158: Keys.n0,
    156: Keys.np1,
    153: Keys.np2,
    155: Keys.np3,
    150: Keys.np4,
    157: Keys.np5,
    152: Keys.np6,
    149: Keys.np7,
    151: Keys.np8,
    154: Keys.np9,
}


def cvkey_to_key(cvkeycode):
    """
    Given a cv2 keycode, return the key, which will be a member of the Keys enum.
    :param cvkeycode: The code returned by cv2.waitKey
    :return: A string, one of the members of Keys
    """
    key = _keydict.get(cvkeycode & 0xFF if cvkeycode > 0 else cvkeycode,
                       Keys.UNKNOWN)  # On Mac, keys return codes like 1113938.  Masking with 0xFF limits it to 0-255.
    if key == Keys.UNKNOWN:
        print("Unknown cv2 Key Code: {}".format(cvkeycode))
    return key
