import os
import gradio as gr
from solution import solutions
from sympy.core.numbers import Rational

EN_US = os.getenv("LANG") != "zh_CN.UTF-8"
ZH2EN = {
    "# “注意到”证明法比较大小": "# Compare sizes by 'Note that' proof",
    "比较 e^(m/n) 与 u/v 大小": "Compare e^(m/n) and u/v",
    "分母不能为 0": "The denominator cannot be 0",
    "比较 π^n 与 p/q 大小": "Compare π^n and p/q",
    "比较 e 与 p/q 大小": "Compare e and p/q",
    "比较 π 与 p/q 大小": "Compare π and p/q",
    "#### 证明结果": "#### Proof result",
    "注意到": "Note that",
    "状态栏": "Status",
    "证毕!": "QED",
}


def _L(zh_txt: str):
    return ZH2EN[zh_txt] if EN_US else zh_txt


def generate_md(args_dict, name):
    ans_latex = solutions[name](**args_dict).get_latex_ans()
    return f"{_L('注意到')} $${ans_latex}$$ {_L('证毕!')}"


def float_to_fraction(x):
    x_str = "{0:.10f}".format(x).rstrip("0").rstrip(".")  # 移除小数点后的无效零

    # 检查是否为整数
    if "." not in x_str:
        return Rational(x_str), Rational(1)

    # 分割整数部分和小数部分
    integer_part, decimal_part = x_str.split(".")
    decimal_digits = len(decimal_part)

    # 构造分子和分母
    numerator = int(integer_part + decimal_part)
    denominator = 10**decimal_digits

    # 简化分数
    gcd_value = 1
    a = numerator
    b = denominator
    while b != 0:
        a, b = b, a % b
        gcd_value = a

    p = Rational(numerator // gcd_value)
    q = Rational(denominator // gcd_value)
    return p, q


def infer_pi(p, q):
    status = "Success"
    proof = None
    try:
        if q == 0:
            raise ValueError(_L("分母不能为 0"))

        p, q = float_to_fraction(p / q)
        args_dict = {"p": p, "q": q}
        proof = generate_md(args_dict, "π")

    except Exception as e:
        status = f"{e}"

    return status, proof


def infer_e(p, q):
    status = "Success"
    proof = None
    try:
        if q == 0:
            raise ValueError(_L("分母不能为 0"))

        p, q = float_to_fraction(p / q)
        args_dict = {"p": p, "q": q}
        proof = generate_md(args_dict, "e")

    except Exception as e:
        status = f"{e}"

    return status, proof


def infer_eq(q1, q2, u, v):
    status = "Success"
    proof = None
    try:
        if q2 == 0 or v == 0:
            raise ValueError(_L("分母不能为 0"))

        q1, q2 = float_to_fraction(q1 / q2)
        u, v = float_to_fraction(u / v)
        args_dict = {"q1": q1, "q2": q2, "u": u, "v": v}
        proof = generate_md(args_dict, "e^q")

    except Exception as e:
        status = f"{e}"

    return status, proof


def infer_pin(n: int, p, q):
    status = "Success"
    proof = None
    try:
        if q == 0:
            raise ValueError(_L("分母不能为 0"))

        p, q = float_to_fraction(p / q)
        args_dict = {"n": n, "p": p, "q": q}
        proof = generate_md(args_dict, "π^n")

    except Exception as e:
        status = f"{e}"

    return status, proof


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    for file_name in os.listdir("solutions"):
        if not file_name.endswith(".py"):
            continue

        __import__(f"solutions.{file_name[:-3]}")

    with gr.Blocks() as demo:
        gr.Markdown(_L("# “注意到”证明法比较大小"))
        with gr.Tabs():
            with gr.TabItem("π"):
                gr.Interface(
                    fn=infer_pi,
                    inputs=[
                        gr.Number(label="p", value=314),
                        gr.Number(label="q", value=100),
                    ],
                    outputs=[
                        gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                        gr.Markdown(
                            value=_L("#### 证明结果"),
                            buttons=["copy"],
                            container=True,
                            min_height=122,
                        ),
                    ],
                    title=_L("比较 π 与 p/q 大小"),
                    flagging_mode="never",
                )

            with gr.TabItem("e"):
                gr.Interface(
                    fn=infer_e,
                    inputs=[
                        gr.Number(label="p", value=2718),
                        gr.Number(label="q", value=1000),
                    ],
                    outputs=[
                        gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                        gr.Markdown(
                            value=_L("#### 证明结果"),
                            buttons=["copy"],
                            container=True,
                            min_height=122,
                        ),
                    ],
                    title=_L("比较 e 与 p/q 大小"),
                    flagging_mode="never",
                )

            with gr.TabItem("e^q"):
                gr.Interface(
                    fn=infer_eq,
                    inputs=[
                        gr.Number(label="m", value=3),
                        gr.Number(label="n", value=4),
                        gr.Number(label="u", value=2117),
                        gr.Number(label="v", value=1000),
                    ],
                    outputs=[
                        gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                        gr.Markdown(
                            value=_L("#### 证明结果"),
                            buttons=["copy"],
                            container=True,
                            min_height=122,
                        ),
                    ],
                    title=_L("比较 e^(m/n) 与 u/v 大小"),
                    flagging_mode="never",
                )

            with gr.TabItem("π^n"):
                gr.Interface(
                    fn=infer_pin,
                    inputs=[
                        gr.Number(label="n", value=3, step=1),
                        gr.Number(label="p", value=31),
                        gr.Number(label="q", value=1),
                    ],
                    outputs=[
                        gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                        gr.Markdown(
                            value=_L("#### 证明结果"),
                            buttons=["copy"],
                            container=True,
                            min_height=122,
                        ),
                    ],
                    title=_L("比较 π^n 与 p/q 大小"),
                    flagging_mode="never",
                )

    demo.launch(css="#gradio-share-link-button-0 { display: none; }", ssr_mode=False)
