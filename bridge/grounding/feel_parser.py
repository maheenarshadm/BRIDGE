"""Parser for the FEEL subset used in the decision tables: unary tests of input entries and simple expressions."""
import re


class UnsupportedFeelConstruct(Exception):
    pass


TOKEN_RE = re.compile(r"""
    \s*(?:
        (?P<STRING>"(?:[^"\\]|\\.)*")
      | (?P<NUMBER>-?\d+\.\d+|-?\d+)
      | (?P<OP><=|>=|!=|\.\.|[<>=+\-*/(),])
      | (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)
      | (?P<LBRACKET>\[)
      | (?P<RBRACKET>\])
    )
""", re.VERBOSE)

KEYWORDS = {'true', 'false', 'not', 'if', 'then', 'else', 'for', 'in', 'where',
            'return', 'and', 'or'}


def tokenize(text):
    tokens = []
    pos = 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            if text[pos:].strip() == '':
                break
            raise UnsupportedFeelConstruct(f"unrecognized token at {pos!r} in {text!r}")
        pos = m.end()
        kind = m.lastgroup
        val = m.group(kind)
        tokens.append((kind, val))
    return tokens


class _Parser:
    def __init__(self, text):
        self.text = text
        self.tokens = tokenize(text)
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else (None, None)

    def advance(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, val):
        kind, v = self.advance()
        if v != val:
            raise UnsupportedFeelConstruct(f"expected {val!r}, got {v!r} in {self.text!r}")

    def at_end(self):
        return self.i >= len(self.tokens)

    def parse_expression(self):
        kind, val = self.peek()
        if val == 'if':
            return self._parse_if()
        return self._parse_comparison()

    def _parse_if(self):
        self.expect('if')
        cond = self._parse_comparison()
        self.expect('then')
        then_e = self.parse_expression()
        self.expect('else')
        else_e = self.parse_expression()
        return {'op': 'if', 'cond': cond, 'then': then_e, 'else': else_e}

    _COMPARISON_OPS = {'=', '!=', '<', '<=', '>', '>='}

    def _parse_comparison(self):
        left = self._parse_additive()
        kind, val = self.peek()
        if val in self._COMPARISON_OPS:
            self.advance()
            right = self._parse_additive()
            return {'op': val, 'left': left, 'right': right}
        return left

    def _parse_additive(self):
        left = self._parse_multiplicative()
        while self.peek()[1] in ('+', '-'):
            op = self.advance()[1]
            right = self._parse_multiplicative()
            left = {'op': op, 'left': left, 'right': right}
        return left

    def _parse_multiplicative(self):
        left = self._parse_primary()
        while self.peek()[1] in ('*', '/'):
            op = self.advance()[1]
            right = self._parse_primary()
            left = {'op': op, 'left': left, 'right': right}
        return left

    def _parse_primary(self):
        kind, val = self.peek()
        if kind == 'NUMBER':
            self.advance()
            return {'kind': 'literal', 'value': float(val) if '.' in val else int(val), 'type': 'number'}
        if kind == 'STRING':
            self.advance()
            return {'kind': 'literal', 'value': val[1:-1], 'type': 'string'}
        if val in ('true', 'false'):
            self.advance()
            return {'kind': 'literal', 'value': val == 'true', 'type': 'boolean'}
        if val == '(':
            self.advance()
            inner = self.parse_expression()
            self.expect(')')
            return inner
        if kind == 'IDENT':
            self.advance()
            if self.peek()[1] == '(':
                return self._parse_call(val)
            return {'kind': 'variable', 'ref': val}
        raise UnsupportedFeelConstruct(f"unexpected token {val!r} in {self.text!r}")

    def _parse_call(self, name):
        self.expect('(')
        args = []
        if self.peek()[1] != ')':
            args.append(self._parse_call_arg())
            while self.peek()[1] == ',':
                self.advance()
                args.append(self._parse_call_arg())
        self.expect(')')
        return {'kind': 'call', 'name': name, 'args': args}

    def _parse_call_arg(self):
        start_i = self.i
        try:
            expr = self.parse_expression()
            if self.peek()[1] != 'for':
                return expr
        except UnsupportedFeelConstruct:
            pass
        self.i = start_i
        depth = 0
        raw_tokens = []
        while not self.at_end():
            kind, val = self.peek()
            if val == '(':
                depth += 1
            elif val == ')':
                if depth == 0:
                    break
                depth -= 1
            elif val == ',' and depth == 0:
                break
            raw_tokens.append(val)
            self.advance()
        free_vars = sorted({t for t in raw_tokens
                             if re.match(r'^[A-Za-z_]', t) and t not in KEYWORDS})
        return {'kind': 'opaque_formula', 'feel_text': ' '.join(raw_tokens),
                'free_variables': free_vars}


def parse_expression(text):
    p = _Parser(text)
    node = p.parse_expression()
    if not p.at_end():
        raise UnsupportedFeelConstruct(f"trailing tokens after parse: {text!r}")
    return node


def _literal_from_token(kind, val):
    if kind == 'NUMBER':
        return {'kind': 'literal', 'value': float(val) if '.' in val else int(val), 'type': 'number'}
    if kind == 'STRING':
        return {'kind': 'literal', 'value': val[1:-1], 'type': 'string'}
    if val in ('true', 'false'):
        return {'kind': 'literal', 'value': val == 'true', 'type': 'boolean'}
    if kind == 'IDENT':
        return {'kind': 'variable', 'ref': val}
    raise UnsupportedFeelConstruct(f"not a literal/identifier: {val!r}")


def _parse_bound(p):
    kind, val = p.advance()
    if kind == 'IDENT' and p.peek()[1] == '(':
        return p._parse_call(val)
    return _literal_from_token(kind, val)


def parse_unary_test(text, column_var):
    text = (text or '').strip()
    if text in ('', '-'):
        return None

    left = {'kind': 'variable', 'ref': column_var}
    p = _Parser(text)

    kind, val = p.peek()

    if val == 'not':
        p.advance()
        p.expect('(')
        inner = _parse_test_body(p, left)
        p.expect(')')
        if not p.at_end():
            raise UnsupportedFeelConstruct(f"trailing tokens after not(...): {text!r}")
        return {'op': 'not', 'clause': inner}

    if val in ('<', '<=', '>', '>=', '!='):
        p.advance()
        right = _parse_bound(p)
        if not p.at_end():
            raise UnsupportedFeelConstruct(f"trailing tokens: {text!r}")
        return {'op': val, 'left': left, 'right': right}

    node = _parse_test_body(p, left)
    if not p.at_end():
        raise UnsupportedFeelConstruct(f"trailing tokens: {text!r}")
    return node


def _parse_test_body(p, left):
    if p.peek()[1] == '[':
        p.advance()
        low = _parse_bound(p)
        p.expect('..')
        high = _parse_bound(p)
        p.expect(']')
        return {'op': 'between', 'left': left, 'low': low, 'high': high}

    values = [_parse_bound(p)]
    while p.peek()[1] == ',':
        p.advance()
        values.append(_parse_bound(p))

    if len(values) == 1:
        return {'op': '=', 'left': left, 'right': values[0]}
    return {'op': 'in', 'left': left, 'values': values}

