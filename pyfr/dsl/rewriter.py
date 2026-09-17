from pyfr.dsl.codegen import CodeGenerator
from pyfr.dsl.lexer import Lexer
from pyfr.dsl.nodes import (DslVar, Float, Index, Var, VarDecl, map_ast,
                            unwrap_index)
from pyfr.dsl.parser import Parser
from pyfr.dsl.simplifier import Simplifier


def parse_expr(code, lexer=Lexer):
    p = Parser(lexer(code).tokenise())
    expr = p.expression(0)

    # Trailing tokens would otherwise be silently discarded
    if (tok := p.peek()).kind != 'EOF':
        raise SyntaxError(f'Unexpected {tok.value!r} in expression: {code}')

    return expr


def parse_stmt(code):
    return Parser(Lexer(code).tokenise()).statement()


def parse_program(code, types=()):
    return Parser(Lexer(code, types=types).tokenise()).parse()


def rename_vars(ast, renames):
    def rename(node):
        match node:
            case Var(name) | DslVar(name) if name in renames:
                return renames[name]
            case VarDecl(c, vt, n, sz, init) if n in renames:
                r = renames[n]

                # Only a plain variable can stand in for a declaration
                if isinstance(r, Var):
                    if sz:
                        sz = [rename(s) for s in sz]

                    init = rename(init) if init else None
                    return VarDecl(c, vt, r.name, sz, init)
                else:
                    return map_ast(node, rename)
            case _:
                return map_ast(node, rename)

    return rename(ast)


def rewrite_exprs(exprs, rules, coords={}, subs={}):
    coords = {c: parse_expr(v) for c, v in coords.items()}
    rsubs = coords | {k: parse_expr(v) for k, v in subs.items()}
    rules = {k: rename_vars(parse_expr(v), rsubs) for k, v in rules.items()}
    exprs = {k: parse_expr(v) for k, v in exprs.items()}

    csubs = [(coords[c], r) for c, r in rules.items() if c in coords]

    def coord(node):
        for pat, r in csubs:
            if node == pat:
                return r

        return map_ast(node, coord)

    rexprs = {k: coord(v) for k, v in exprs.items()}
    rexprs |= {k: rename_vars(r, rexprs) for k, r in rules.items()
               if k in exprs}

    gen, simp = CodeGenerator().generate, Simplifier().simplify_fully
    return {k: gen(simp(r)) for k, r in rexprs.items() if r != exprs[k]}


def fold_indices(ast):
    simp = Simplifier()

    # Fold constant arithmetic inside array indices
    def fold(node):
        node = map_ast(node, fold)
        if isinstance(node, Index):
            return Index(node.array, simp.simplify_fully(node.index))
        else:
            return node

    return fold(ast)


class Rewriter:
    def __init__(self, rules):
        self.rules = rules

    def transform(self, node, args):
        mapped = map_ast(node, lambda n: self.transform(n, args))

        # Extract the name and indices, then look up a matching arg
        n, ix = unwrap_index(mapped)
        if n is not None and n in args:
            arg = args[n]
            for r in self.rules:
                if (result := r(n, ix, arg)) is not None:
                    return result

        return mapped


class FloatSuffixer:
    def transform(self, node):
        if isinstance(node, Float) and not node.suffix:
            return Float(node.value, 'f', node.ishex)
        else:
            return map_ast(node, self.transform)
