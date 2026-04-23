from sympy.abc import a, b, c, x
from output_util import to_latex, sign2cmp
from sgntools import sq_func_sgn
from solution import *


class PiIntegrate(GetIntegrateFromData):
    # sympy 算力不够，以下由 MMA 算出
    # PiIF[n_] :=
    #  Collect[ Collect[
    #    PowerExpand[ Expand[
    #      Integrate[x^n*(1 - x)^n*(a + b x + c x^2)/(x^2 + 1),
    #                {x, 0, 1}]
    #    ]],
    #    Log[2]], Pi]
    data = {
        1: {
            "pi_term": (a - b - c) / 4,
            "log2_term": (a + b - c) / 2,
            CONSTANT_TERM_KEY: -a + b / 2 + 7 * c / 6,
        },
        2: {
            "pi_term": -b / 2,
            "log2_term": a - c,
            CONSTANT_TERM_KEY: (-2 * a / 3 + 19 * b / 12 + 7 * c / 10),
        },
        3: {
            "pi_term": (-a - b + c) / 2,
            "log2_term": a - b - c,
            CONSTANT_TERM_KEY: 53 * a / 60 + 34 * b / 15 - 92 * c / 105,
        },
        4: {
            "pi_term": -a + c,
            "log2_term": 2 * b,
            CONSTANT_TERM_KEY: 22 * a / 7 + 233 * b / 168 - 1979 * c / 630,
        },
        5: {
            "pi_term": -a + b + c,
            "log2_term": 2 * (-a - b + c),
            CONSTANT_TERM_KEY: 11411 * a / 2520 - 4423 * b / 2520 - 41837 * c / 9240,
        },
        6: {
            "pi_term": 2 * b,
            "log2_term": -4 * a + 4 * c,
            CONSTANT_TERM_KEY: (38429 * a) / 13860
            - (174169 * b) / 27720
            - (35683 * c) / 12870,
        },
        7: {
            "pi_term": 2 * (a + b - c),
            "log2_term": 4 * (-a + b + c),
            CONSTANT_TERM_KEY: -((421691 * a) / 120120)
            - (407917 * b) / 45045
            + (31627 * c) / 9009,
        },
        8: {
            "pi_term": 4 * (a - c),
            "log2_term": 8 * b,
            CONSTANT_TERM_KEY: -(188684 * a) / 15015
            - (1332173 * b) / 240240
            + (3849155 * c) / 306306,
        },
        9: {
            "pi_term": 4 * (a - b - c),
            "log2_term": 8 * (-c + a + b),
            CONSTANT_TERM_KEY: -((17069771 * a) / 942480)
            + (86025349 * b) / 12252240
            + (4216233689 * c) / 232792560,
        },
        10: {
            "pi_term": -8 * b,
            "log2_term": 16 * (a - c),
            CONSTANT_TERM_KEY: -((1290876029 * a) / 116396280)
            + (325039733 * b) / 12932920
            + (117352369 * c) / 10581480,
        },
        11: {
            "pi_term": 8 * (-a - b + c),
            "log2_term": 16 * (a - b - c),
            CONSTANT_TERM_KEY: (817240769 * a) / 58198140
            + (4216233641 * b) / 116396280
            - (37593075209 * c) / 2677114440,
        },
        12: {
            "pi_term": -16 * a + 16 * c,
            "log2_term": -32 * b,
            CONSTANT_TERM_KEY: (431302721 * a) / 8580495
            + (9135430531 * b) / 411863760
            - (5902037233 * c) / 117417300,
        },
    }

    def get_integrate_args(self, n):
        return x**n * (1 - x) ** n * (a + b * x + c * x**2) / (1 + x**2), (x, 0, 1)


class PiSolution(Solution):
    def __init__(self, p, q):
        self.p = p
        self.q = q
        if self.q == 0:
            raise BadInput()

        self.get_integrate = PiIntegrate()
        self.symbols = (a, b, c)
        self.integrate_result_classes_eq = {
            "pi_term": q,
            "log2_term": 0,
            CONSTANT_TERM_KEY: -p,
        }
        # check
        self.check_sgn = sq_func_sgn

    @staticmethod
    def get_tries_args():
        return PiIntegrate.data.keys()

    def get_latex_ans(self):
        try_arg, symbol_val, sgn = self.try_times()
        print(f"{(try_arg, symbol_val, sgn)=}")
        if try_arg is None:
            return None

        p = to_latex(self.p)
        q = to_latex(self.q, is_coeff=True)
        I = self.get_integrate.get_latex(try_arg, symbol_val)
        return rf"{q}\pi-{p} = {I} {sign2cmp[sgn]} 0"


register("π", PiSolution, top=True)

if __name__ == "__main__":
    while True:
        p = int(input(">>> "))
        q = int(input("  / "))
        print(PiSolution(p, q).get_latex_ans())
