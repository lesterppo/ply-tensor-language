"""
Ply Parser — Recursive descent parser for the tensor-native language.
Converts token stream into a static AST with no control flow constructs.

Operator precedence (lowest to highest):
  ?:        ternary conditional (mask-based branching)
  |         element-wise logical OR
  &         element-wise logical AND
  == != > < >= <=   comparison
  + -       element-wise add/sub
  * / @     element-wise mul/div, matrix multiply
  - !       unary negate, logical not
  () []     grouping, slicing
"""
from tokens import Token, TokenType, tokenize
from ast_nodes import *


class ParseError(Exception):
    def __init__(self, msg: str, token: Token):
        super().__init__(f"Line {token.line}, col {token.col}: {msg} (got '{token.value}')")
        self.token = token


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    @property
    def current(self) -> Token:
        return self.tokens[self.pos]

    @property
    def peek(self) -> Token:
        return self.tokens[min(self.pos + 1, len(self.tokens) - 1)]

    def advance(self) -> Token:
        tok = self.current
        self.pos += 1
        return tok

    def expect(self, tt: TokenType) -> Token:
        if self.current.type != tt:
            raise ParseError(f"Expected {tt.name}", self.current)
        return self.advance()

    def match(self, *tts: TokenType) -> bool:
        return self.current.type in tts

    # ── Top level ────────────────────────────────────────

    def parse_program(self) -> Program:
        bindings = []
        result = None
        while not self.match(TokenType.EOF):
            while self.match(TokenType.NEWLINE):
                self.advance()
            if self.match(TokenType.EOF):
                break

            # Detect bare expression (no :=) — must be the final result
            if self._is_bare_expression():
                result = self.parse_expression()
                while self.match(TokenType.NEWLINE):
                    self.advance()
                if not self.match(TokenType.EOF):
                    raise ParseError(
                        "Bare expression only allowed as the final line", self.current
                    )
                break

            bindings.append(self.parse_binding())
            if self.match(TokenType.NEWLINE):
                self.advance()
            elif not self.match(TokenType.EOF):
                raise ParseError("Expected newline after binding", self.current)

        return Program(bindings=bindings, result=result or (bindings[-1].value if bindings else None))

    def _is_bare_expression(self) -> bool:
        """Check if current line is a bare expression (no :=).
        Look ahead: ID followed by ( or [ or operator or ->, or a literal/function call.
        """
        if self.match(TokenType.INT, TokenType.FLOAT, TokenType.STRING,
                       TokenType.LPAREN, TokenType.LBRACK,
                       TokenType.MINUS, TokenType.NOT):
            return True
        if self.match(TokenType.ID):
            # Check if the next token is BIND or something else
            # Save state
            saved = self.pos
            self.advance()
            is_bare = not self.match(TokenType.BIND)
            self.pos = saved
            return is_bare
        return False

    def parse_binding(self) -> Binding:
        name = self.expect(TokenType.ID).value
        self.expect(TokenType.BIND)
        value = self.parse_expression()
        return Binding(name, value)

    # ── Expressions (precedence climbing) ────────────────

    def parse_expression(self) -> Expr:
        return self.parse_ternary()

    def parse_ternary(self) -> Expr:
        left = self.parse_logical_or()
        if self.match(TokenType.QUESTION):
            self.advance()
            true_val = self.parse_expression()
            self.expect(TokenType.COLON)
            false_val = self.parse_expression()
            return Ternary(left, true_val, false_val)
        return left

    def parse_logical_or(self) -> Expr:
        return self._left_assoc_binary(
            self.parse_logical_and,
            TokenType.OR
        )

    def parse_logical_and(self) -> Expr:
        return self._left_assoc_binary(
            self.parse_comparison,
            TokenType.AND
        )

    def parse_comparison(self) -> Expr:
        return self._left_assoc_binary(
            self.parse_addsub,
            TokenType.EQ, TokenType.NE, TokenType.LT, TokenType.GT,
            TokenType.LE, TokenType.GE
        )

    def parse_addsub(self) -> Expr:
        return self._left_assoc_binary(
            self.parse_muldiv,
            TokenType.PLUS, TokenType.MINUS
        )

    def parse_muldiv(self) -> Expr:
        return self._left_assoc_binary(
            self.parse_unary,
            TokenType.STAR, TokenType.SLASH, TokenType.AT
        )

    def _left_assoc_binary(self, inner, *ops: TokenType) -> Expr:
        node = inner()
        while self.match(*ops):
            op = self.advance().value
            right = inner()
            node = BinOp(op, node, right)
        return node

    def parse_unary(self) -> Expr:
        if self.match(TokenType.MINUS):
            self.advance()
            return UnOp('-', self.parse_unary())
        if self.match(TokenType.NOT):
            self.advance()
            return UnOp('!', self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> Expr:
        tok = self.current

        if tok.type == TokenType.INT:
            self.advance()
            return Number(float(tok.value))
        if tok.type == TokenType.FLOAT:
            self.advance()
            return Number(float(tok.value))
        if tok.type == TokenType.STRING:
            self.advance()
            return String(tok.value[1:-1])  # strip quotes
        if tok.type == TokenType.LBRACK:
            return self.parse_list_literal()
        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return self._attach_slice_or_call(expr)

        if tok.type == TokenType.ID:
            return self._parse_id_or_call()

        raise ParseError("Unexpected token", tok)

    def _parse_id_or_call(self) -> Expr:
        name = self.advance().value
        if self.match(TokenType.LPAREN):
            return self._parse_call_tail(name)
        node: Expr = Var(name)
        return self._attach_slice_or_call(node)

    def _parse_call_tail(self, name: str) -> Call:
        self.advance()  # (
        args = self.parse_arglist()
        self.expect(TokenType.RPAREN)
        node: Expr = Call(name, args)
        return self._attach_slice_or_call(node)

    def _attach_slice_or_call(self, node: Expr) -> Expr:
        """Handle postfix [] slicing and .attr on any expression."""
        while True:
            if self.match(TokenType.LBRACK):
                node = self.parse_slice_tail(node)
            elif self.match(TokenType.DOT):
                self.advance()
                attr = self.expect(TokenType.ID).value
                node = Call(attr, [node])
            else:
                break
        return node

    def parse_slice_tail(self, tensor: Expr) -> Slice:
        self.advance()  # [
        slices = self.parse_slicelist()
        self.expect(TokenType.RBRACK)
        return Slice(tensor, slices)

    def parse_slicelist(self) -> List[SliceSpec]:
        specs = []
        specs.append(self.parse_slice_spec())
        while self.match(TokenType.COMMA):
            self.advance()
            specs.append(self.parse_slice_spec())
        return specs

    def parse_slice_spec(self) -> SliceSpec:
        """Parse start:end:step or bare expression."""
        if self.match(TokenType.COLON):
            # :end  or ::step
            self.advance()
            if self.match(TokenType.COLON, TokenType.COMMA, TokenType.RBRACK):
                return SliceSpec(None, None, None)  # bare :
            end = self._slice_expr()
            if self.match(TokenType.COLON):
                self.advance()
                step = self._slice_expr()
                return SliceSpec(None, end, step)
            return SliceSpec(None, end, None)

        start = self._slice_expr()
        if not self.match(TokenType.COLON):
            return SliceSpec(start, None, None)  # single index

        self.advance()  # consume :

        if self.match(TokenType.COLON, TokenType.COMMA, TokenType.RBRACK):
            return SliceSpec(start, None, None)

        end = self._slice_expr()
        if self.match(TokenType.COLON):
            self.advance()
            step = self._slice_expr()
            return SliceSpec(start, end, step)
        return SliceSpec(start, end, None)

    def _slice_expr(self) -> Optional[Expr]:
        if self.match(TokenType.COLON, TokenType.COMMA, TokenType.RBRACK):
            return None
        return self.parse_expression()

    def parse_list_literal(self) -> ListLiteral:
        self.advance()  # [
        elements = self.parse_arglist()
        self.expect(TokenType.RBRACK)
        return ListLiteral(elements)

    def parse_arglist(self) -> List[Expr]:
        args = []
        if not self.match(TokenType.RPAREN, TokenType.RBRACK):
            args.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                self.advance()
                args.append(self.parse_expression())
        return args


def parse(source: str) -> Program:
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_program()
