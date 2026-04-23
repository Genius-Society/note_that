from sympy import *
from sympy.abc import a, b, x
from output_util import to_latex, sign2cmp, to_latexes
from sgntools import lin_func_sgn
from solution import *

_data = {
    (1, 2): {"constant_term": a - 2 * b / 3, "pin_term": -a / 4 + b / 4},
    (1, 4): {"constant_term": -2 * a / 3 + 13 * b / 15, "pin_term": a / 4 - b / 4},
    (1, 6): {"constant_term": 13 * a / 15 - 76 * b / 105, "pin_term": -a / 4 + b / 4},
    (1, 8): {"constant_term": -76 * a / 105 + 263 * b / 315, "pin_term": a / 4 - b / 4},
    (1, 10): {
        "constant_term": 263 * a / 315 - 2578 * b / 3465,
        "pin_term": -a / 4 + b / 4,
    },
    (1, 12): {
        "constant_term": -2578 * a / 3465 + 36979 * b / 45045,
        "pin_term": a / 4 - b / 4,
    },
    (2, 3): {"constant_term": a / 4 - 3 * b / 16, "pin_term": -a / 48 + b / 48},
    (2, 5): {"constant_term": -3 * a / 16 + 31 * b / 144, "pin_term": a / 48 - b / 48},
    (2, 7): {
        "constant_term": 31 * a / 144 - 115 * b / 576,
        "pin_term": -a / 48 + b / 48,
    },
    (2, 9): {
        "constant_term": -115 * a / 576 + 3019 * b / 14400,
        "pin_term": a / 48 - b / 48,
    },
    (2, 11): {
        "constant_term": 3019 * a / 14400 - 973 * b / 4800,
        "pin_term": -a / 48 + b / 48,
    },
    (2, 13): {
        "constant_term": -973 * a / 4800 + 48877 * b / 235200,
        "pin_term": a / 48 - b / 48,
    },
    (3, 2): {"constant_term": 2 * a - 52 * b / 27, "pin_term": -a / 16 + b / 16},
    (3, 4): {
        "constant_term": -52 * a / 27 + 6554 * b / 3375,
        "pin_term": a / 16 - b / 16,
    },
    (3, 6): {
        "constant_term": 6554 * a / 3375 - 2241272 * b / 1157625,
        "pin_term": -a / 16 + b / 16,
    },
    (3, 8): {
        "constant_term": -2241272 * a / 1157625 + 60600094 * b / 31255875,
        "pin_term": a / 16 - b / 16,
    },
    (3, 10): {
        "constant_term": 60600094 * a / 31255875 - 80596213364 * b / 41601569625,
        "pin_term": -a / 16 + b / 16,
    },
    (3, 12): {
        "constant_term": -80596213364 * a / 41601569625
        + 177153083899958 * b / 91398648466125,
        "pin_term": a / 16 - b / 16,
    },
    (4, 3): {
        "constant_term": 3 * a / 8 - 45 * b / 128,
        "pin_term": -7 * a / 1920 + 7 * b / 1920,
    },
    (4, 5): {
        "constant_term": -45 * a / 128 + 1231 * b / 3456,
        "pin_term": 7 * a / 1920 - 7 * b / 1920,
    },
    (4, 7): {
        "constant_term": 1231 * a / 3456 - 19615 * b / 55296,
        "pin_term": -7 * a / 1920 + 7 * b / 1920,
    },
    (4, 9): {
        "constant_term": -19615 * a / 55296 + 12280111 * b / 34560000,
        "pin_term": 7 * a / 1920 - 7 * b / 1920,
    },
    (4, 11): {
        "constant_term": 12280111 * a / 34560000 - 4090037 * b / 11520000,
        "pin_term": -7 * a / 1920 + 7 * b / 1920,
    },
    (4, 13): {
        "constant_term": -4090037 * a / 11520000 + 9824498837 * b / 27659520000,
        "pin_term": 7 * a / 1920 - 7 * b / 1920,
    },
    (5, 2): {
        "constant_term": 24 * a - 1936 * b / 81,
        "pin_term": -5 * a / 64 + 5 * b / 64,
    },
    (5, 4): {
        "constant_term": -1936 * a / 81 + 6051944 * b / 253125,
        "pin_term": 5 * a / 64 - 5 * b / 64,
    },
    (5, 6): {
        "constant_term": 6051944 * a / 253125 - 101708947808 * b / 4254271875,
        "pin_term": -5 * a / 64 + 5 * b / 64,
    },
    (5, 8): {
        "constant_term": -101708947808 * a / 4254271875
        + 24715694492344 * b / 1033788065625,
        "pin_term": 5 * a / 64 - 5 * b / 64,
    },
    (5, 10): {
        "constant_term": 24715694492344 * a / 1033788065625
        - 3980462502772918544 * b / 166492601756971875,
        "pin_term": -5 * a / 64 + 5 * b / 64,
    },
    (5, 12): {
        "constant_term": -3980462502772918544 * a / 166492601756971875
        + 1477921859864507412282392 * b / 61817537584151358384375,
        "pin_term": 5 * a / 64 - 5 * b / 64,
    },
    (6, 3): {
        "constant_term": 15 * a / 8 - 945 * b / 512,
        "pin_term": -31 * a / 16128 + 31 * b / 16128,
    },
    (6, 5): {
        "constant_term": -945 * a / 512 + 229955 * b / 124416,
        "pin_term": 31 * a / 16128 - 31 * b / 16128,
    },
    (6, 7): {
        "constant_term": 229955 * a / 124416 - 14713475 * b / 7962624,
        "pin_term": -31 * a / 16128 + 31 * b / 16128,
    },
    (6, 9): {
        "constant_term": -14713475 * a / 7962624 + 45982595359 * b / 24883200000,
        "pin_term": 31 * a / 16128 - 31 * b / 16128,
    },
    (6, 11): {
        "constant_term": 45982595359 * a / 24883200000 - 5109066151 * b / 2764800000,
        "pin_term": -31 * a / 16128 + 31 * b / 16128,
    },
    (6, 13): {
        "constant_term": -5109066151 * a / 2764800000
        + 601081707598999 * b / 325275955200000,
        "pin_term": 31 * a / 16128 - 31 * b / 16128,
    },
}


class PiNIntegrate(GetIntegrateFromData):
    data = _data

    def get_integrate_args(self, args):
        n, m = args
        return x**m * (a + b * x**2) * (log(1 / x)) ** (n - 1) / (1 + x**2), (x, 0, 1)


class PiNSolution(Solution):
    def __init__(self, n, p, q):
        if n < 0:
            n = -n
            p, q = q, p
        self.n = n
        self.p = p
        self.q = q
        if self.q == 0:
            raise BadInput()

        self.get_integrate = PiNIntegrate()
        self.symbols = (a, b)
        self.integrate_result_classes_eq = {
            "pin_term": q,
            CONSTANT_TERM_KEY: -p,
        }
        # check
        self.check_sgn = lin_func_sgn

    def get_tries_args(self):
        for n, m in PiNIntegrate.data.keys():
            if n == self.n:
                yield n, m

    def get_latex_ans(self):
        try_arg, symbol_val, sgn = self.try_times()
        print(f"{(try_arg, symbol_val, sgn)=}")
        if try_arg is None:
            return None

        p, n = to_latexes(self.p, self.n)
        q = to_latex(self.q, is_coeff=True)
        I = self.get_integrate.get_latex(try_arg, symbol_val)
        return rf"{q}\pi^{n}-{p} = {I} {sign2cmp[sgn]} 0"


register("π^n", PiNSolution)


if __name__ == "__main__":
    print(PiNSolution(3, 31, 1).get_latex_ans())
