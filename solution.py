from typing import Iterable, Callable
from sympy import Eq, Expr, Symbol, linsolve, simplify, latex


class CannotCalculate(Exception):
    pass


class BadInput(Exception):
    pass


CONSTANT_TERM_KEY = "constant_term"


class GetIntegrate:
    integrate_result_classes: dict[str, Expr | None]

    def integrate_and_separate(
        self, integrate_formula: Expr, integrate_args: tuple
    ) -> dict[str, Expr]:
        """
        解积分，分离各项
        """
        raise NotImplementedError(
            "该函数已弃用，请使用 GetIntegrateFromData 类。如果想要使用它预处理，见 pre/old_solution.py"
        )

    def get_integrate_args(self, try_arg):
        """
        :return: integrand_function, (independent_variable, lower_limit, upper_limit)
                被积函数, (自变量, 下限, 上限)
        """
        raise NotImplementedError()

    def tries(self, try_arg):
        """
        :return: {term_name: coefficient_of_the_term}
                {该项的名称: 该项系数}
        """
        return self.integrate_and_separate(*self.get_integrate_args(try_arg))

    def get_latex(self, try_arg, subs):
        expr, args = self.get_integrate_args(try_arg)
        expr = simplify(expr).subs(subs)
        sym, low, high = args
        return r"\int_{%s}^{%s} {%s} \mathrm{d} {%s}" % (
            low,
            high,
            latex(simplify(expr)),
            sym,
        )


class GetIntegrateFromData(GetIntegrate):
    data: dict[object, dict[str, Expr]]

    def tries(self, try_arg):
        return self.data[try_arg]


class Solution:
    get_integrate: GetIntegrate
    # gui: Pattern

    get_tries_args: Callable[[], Iterable[Expr]]
    """
    Generate try_arg for each trial
    生成每一次尝试的 try_arg
    """

    symbols: tuple[Symbol, ...]
    """
    The undetermined variable to solve.
    要求的待定系数
    """

    integrate_result_classes_eq: dict[str, Expr]
    """
    {term_name: coefficient}
    The value to which each term's coefficient is equal.
    每个项系数分别要等于的值
    """

    check_sgn: Callable
    """
    Input: unpacked solution of undetermined variables; 
    Output: 0 (cannot determine), 1 or -1 (can determine; 1 and -1 are interchangeable for greater or less than).
    Module sgntools provides lin_func_sgn and sq_func_sgn. Call help() for further details.
    
    输入：解包的待定系数的值列表；
    输出: 0(无法确定), 1或-1(可以确定, 1和-1哪个代表大于、哪个代表小于, 是可以互换的)
    sgntools 模块提供了 lin_func_sgn 和 sq_func_sgn 函数，详见 help()
    """
    # check_sgn: Callable[?, 1 | 0 | -1]

    def get_symbols(self, separate_result):
        """凑积分结果系数"""
        system = [
            Eq(int_term, self.integrate_result_classes_eq[key])
            for key, int_term in separate_result.items()
        ]
        print(f"{system=}")
        return linsolve(system, *self.symbols)

    def try_times(self) -> tuple[Expr, dict, int] | tuple[None, None, None]:
        for try_arg in self.get_tries_args():
            separate = self.get_integrate.tries(try_arg)
            for symbol_solution in self.get_symbols(separate):
                if sgn := self.check_sgn(*symbol_solution):
                    return try_arg, dict(zip(self.symbols, symbol_solution)), sgn

        return None, None, None

    def get_latex_ans(self):
        """
        :return: None if it can't be solved; Otherwise, LaTeX (without "$$")
        """
        raise NotImplementedError()


# 插件注册（历史遗留问题）
solutions = {}
solution_sort = []


def register(name, solution, top=False):
    solutions[name] = solution
    if top:
        solution_sort.insert(0, name)
    else:
        solution_sort.append(name)
