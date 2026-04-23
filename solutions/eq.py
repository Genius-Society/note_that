from sympy import *
from sympy.abc import *
from output_util import to_latex, sign2cmp, to_latexes
from sgntools import lin_func_sgn
from solution import *

_data = {
    (1,): {
        "constant_term": a / q**2 + 2 * a / q**3 - 2 * b / q**3 - 6 * b / q**4,
        "eq_term": a / q**2 - 2 * a / q**3 + b / q**2 - 4 * b / q**3 + 6 * b / q**4,
    },
    (2,): {
        "constant_term": -2 * a / q**3
        - 12 * a / q**4
        - 24 * a / q**5
        + 6 * b / q**4
        + 48 * b / q**5
        + 120 * b / q**6,
        "eq_term": 2 * a / q**3
        - 12 * a / q**4
        + 24 * a / q**5
        + 2 * b / q**3
        - 18 * b / q**4
        + 72 * b / q**5
        - 120 * b / q**6,
    },
    (3,): {
        "constant_term": 6 * a / q**4
        + 72 * a / q**5
        + 360 * a / q**6
        + 720 * a / q**7
        - 24 * b / q**5
        - 360 * b / q**6
        - 2160 * b / q**7
        - 5040 * b / q**8,
        "eq_term": 6 * a / q**4
        - 72 * a / q**5
        + 360 * a / q**6
        - 720 * a / q**7
        + 6 * b / q**4
        - 96 * b / q**5
        + 720 * b / q**6
        - 2880 * b / q**7
        + 5040 * b / q**8,
    },
    (4,): {
        "constant_term": -24 * a / q**5
        - 480 * a / q**6
        - 4320 * a / q**7
        - 20160 * a / q**8
        - 40320 * a / q**9
        + 120 * b / q**6
        + 2880 * b / q**7
        + 30240 * b / q**8
        + 161280 * b / q**9
        + 362880 * b / q**10,
        "eq_term": 24 * a / q**5
        - 480 * a / q**6
        + 4320 * a / q**7
        - 20160 * a / q**8
        + 40320 * a / q**9
        + 24 * b / q**5
        - 600 * b / q**6
        + 7200 * b / q**7
        - 50400 * b / q**8
        + 201600 * b / q**9
        - 362880 * b / q**10,
    },
    (5,): {
        "constant_term": 120 * a / q**6
        + 3600 * a / q**7
        + 50400 * a / q**8
        + 403200 * a / q**9
        + 1814400 * a / q**10
        + 3628800 * a / q**11
        - 720 * b / q**7
        - 25200 * b / q**8
        - 403200 * b / q**9
        - 3628800 * b / q**10
        - 18144000 * b / q**11
        - 39916800 * b / q**12,
        "eq_term": 120 * a / q**6
        - 3600 * a / q**7
        + 50400 * a / q**8
        - 403200 * a / q**9
        + 1814400 * a / q**10
        - 3628800 * a / q**11
        + 120 * b / q**6
        - 4320 * b / q**7
        + 75600 * b / q**8
        - 806400 * b / q**9
        + 5443200 * b / q**10
        - 21772800 * b / q**11
        + 39916800 * b / q**12,
    },
    (6,): {
        "constant_term": -720 * a / q**7
        - 30240 * a / q**8
        - 604800 * a / q**9
        - 7257600 * a / q**10
        - 54432000 * a / q**11
        - 239500800 * a / q**12
        - 479001600 * a / q**13
        + 5040 * b / q**8
        + 241920 * b / q**9
        + 5443200 * b / q**10
        + 72576000 * b / q**11
        + 598752000 * b / q**12
        + 2874009600 * b / q**13
        + 6227020800 * b / q**14,
        "eq_term": 720 * a / q**7
        - 30240 * a / q**8
        + 604800 * a / q**9
        - 7257600 * a / q**10
        + 54432000 * a / q**11
        - 239500800 * a / q**12
        + 479001600 * a / q**13
        + 720 * b / q**7
        - 35280 * b / q**8
        + 846720 * b / q**9
        - 12700800 * b / q**10
        + 127008000 * b / q**11
        - 838252800 * b / q**12
        + 3353011200 * b / q**13
        - 6227020800 * b / q**14,
    },
}


class EQIntegrate(GetIntegrateFromData):
    data = _data

    def get_integrate_args(self, try_arg):
        n, q = try_arg
        return x**n * (1 - x) ** n * (a + b * x) * e ** (q * x), (x, 0, 1)

    def tries(self, try_arg):
        n, q_value = try_arg
        return {key: expr.subs(q, q_value) for key, expr in self.data[(n,)].items()}


class EQSolution(Solution):
    def __init__(self, q1, q2, u, v):
        if q2 == 0 or v == 0:
            raise BadInput()

        self.q = q1 / q2  # 输入都是 Rational 类型的，可以不损失精度相除
        self.u = u
        self.v = v
        self.get_integrate = EQIntegrate()
        self.symbols = (a, b)
        self.integrate_result_classes_eq = {
            "eq_term": v,
            CONSTANT_TERM_KEY: -u,
        }
        # check
        self.check_sgn = lin_func_sgn

    def get_tries_args(self):
        for (n,) in EQIntegrate.data.keys():
            yield n, self.q

    def get_latex_ans(self):
        try_arg, symbol_val, sgn = self.try_times()
        print(f"{(try_arg, symbol_val, sgn)=}")
        if try_arg is None:
            return None

        u, q = to_latexes(self.u, self.q)
        v = to_latex(self.v, is_coeff=True)
        I = self.get_integrate.get_latex(try_arg, symbol_val)
        return rf"{v}e^{q}-{u} = {I} {sign2cmp[sgn]} 0"


register("e^q", EQSolution)


if __name__ == "__main__":
    print(EQSolution(Rational(2), 1, 7, 1).get_latex_ans())
