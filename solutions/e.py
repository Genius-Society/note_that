from sympy import *
from sympy.abc import a, b, x
from output_util import to_latex, sign2cmp
from sgntools import lin_func_sgn
from solution import *

_data = {
    1: {"constant_term": 3 * a - 8 * b, "e_term": -a + 3 * b},
    2: {"constant_term": -38 * a + 174 * b, "e_term": 14 * a - 64 * b},
    3: {"constant_term": 1158 * a - 7584 * b, "e_term": -426 * a + 2790 * b},
    4: {"constant_term": -65304 * a + 557400 * b, "e_term": 24024 * a - 205056 * b},
    5: {
        "constant_term": 5900520 * a - 62118720 * b,
        "e_term": -2170680 * a + 22852200 * b,
    },
    6: {
        "constant_term": -780827760 * a + 9778048560 * b,
        "e_term": 287250480 * a - 3597143040 * b,
    },
}


class EIntegrate(GetIntegrateFromData):
    # sympy 算力不够，以下由 MMA 算出
    data = _data

    def get_integrate_args(self, n):
        return x**n * (1 - x) ** n * (a + b * x) * exp(x), (x, 0, 1)


class ESolution(Solution):
    def __init__(self, p, q):
        self.p = p
        self.q = q
        if self.q == 0:
            raise BadInput()

        self.get_integrate = EIntegrate()
        self.symbols = (a, b)

        self.integrate_result_classes_eq = {
            "e_term": q,
            CONSTANT_TERM_KEY: -p,
        }
        # check
        self.check_sgn = lin_func_sgn

    @staticmethod
    def get_tries_args():
        return EIntegrate.data.keys()

    def get_latex_ans(self):
        try_arg, symbol_val, sgn = self.try_times()
        print(f"{(try_arg, symbol_val, sgn)=}")
        if try_arg is None:
            return None

        p = to_latex(self.p)
        q = to_latex(self.q, is_coeff=True)
        I = self.get_integrate.get_latex(try_arg, symbol_val)
        return rf"{q}e-{p} = {I} {sign2cmp[sgn]} 0"


register("e", ESolution)


if __name__ == "__main__":
    print(ESolution(25, 9))
