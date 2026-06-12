"""
Ply Tokenizer — Lexer for the tensor-native language.
No keywords. Everything is a token in tensor space.
"""
import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List


class TokenType(Enum):
    ID = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()     # for einsum specs etc.
    BIND = auto()        # :=
    PLUS = auto()        # +
    MINUS = auto()       # -
    STAR = auto()        # *
    SLASH = auto()       # /
    AT = auto()          # @
    LT = auto()          # <
    GT = auto()          # >
    LE = auto()          # <=
    GE = auto()          # >=
    EQ = auto()          # ==
    NE = auto()          # !=
    NOT = auto()         # !
    AND = auto()         # &
    OR = auto()          # |
    QUESTION = auto()    # ?
    COLON = auto()       # :
    ARROW = auto()       # ->
    LPAREN = auto()
    RPAREN = auto()
    LBRACK = auto()
    RBRACK = auto()
    COMMA = auto()
    DOT = auto()
    DOTDOT = auto()      # ..
    COMMENT = auto()
    NEWLINE = auto()
    EOF = auto()


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int


TOKEN_PATTERNS = [
    (r'--[^\n]*',            TokenType.COMMENT),
    (r':=',                  TokenType.BIND),
    (r'<=',                  TokenType.LE),
    (r'>=',                  TokenType.GE),
    (r'==',                  TokenType.EQ),
    (r'!=',                  TokenType.NE),
    (r'->',                  TokenType.ARROW),
    (r'\.\.',                TokenType.DOTDOT),
    (r'<',                   TokenType.LT),
    (r'>',                   TokenType.GT),
    (r'\+',                  TokenType.PLUS),
    (r'-',                   TokenType.MINUS),
    (r'\*',                  TokenType.STAR),
    (r'/',                   TokenType.SLASH),
    (r'@',                   TokenType.AT),
    (r'!',                   TokenType.NOT),
    (r'&',                   TokenType.AND),
    (r'\|',                  TokenType.OR),
    (r'\?',                  TokenType.QUESTION),
    (r':',                   TokenType.COLON),
    (r'\(',                  TokenType.LPAREN),
    (r'\)',                  TokenType.RPAREN),
    (r'\[',                  TokenType.LBRACK),
    (r'\]',                  TokenType.RBRACK),
    (r',',                   TokenType.COMMA),
    (r'\.',                  TokenType.DOT),
    (r'\n',                  TokenType.NEWLINE),
    (r'[ \t\r]+',            None),
    (r"'[^']*'",              TokenType.STRING),
    (r'"[^"]*"',              TokenType.STRING),
    (r'[0-9]+\.[0-9]*([eE][+-]?[0-9]+)?', TokenType.FLOAT),
    (r'[0-9]+[eE][+-]?[0-9]+',            TokenType.FLOAT),
    (r'[0-9]+',              TokenType.INT),
    (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.ID),
]


def tokenize(source: str) -> List[Token]:
    tokens = []
    line, col = 1, 0
    pos = 0

    while pos < len(source):
        matched = False
        for pattern, tok_type in TOKEN_PATTERNS:
            m = re.match(pattern, source[pos:])
            if m:
                text = m.group(0)
                if tok_type is not None and tok_type != TokenType.COMMENT:
                    tokens.append(Token(tok_type, text, line, col))
                if tok_type == TokenType.NEWLINE:
                    line += 1
                    col = 0
                else:
                    col += len(text)
                pos += len(text)
                matched = True
                break
        if not matched:
            raise SyntaxError(
                f"Unexpected character '{source[pos]}' at line {line}, col {col}"
            )

    tokens.append(Token(TokenType.EOF, '', line, col))
    return tokens
