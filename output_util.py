from sympy import latex


def to_latex(expr, is_coeff=False):
    return "" if is_coeff and expr == 1 else f"{{{latex(expr)}}}"


def to_latexes(*args, is_coeff=False):
    return [to_latex(i, is_coeff=is_coeff) for i in args]


sign2cmp = {1: ">", -1: "<"}

sign2cmp_inv = {1: "<", -1: ">"}
