"""Builtin tools.

current_time: server time in Asia/Shanghai as the LLM's conversion baseline.
calculator: safe AST-based arithmetic evaluation (no eval).
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

SERVER_ZONE = ZoneInfo("Asia/Shanghai")


@tool
def current_time() -> str:
    """【强制调用】你自身没有时钟、不知道当前时间,回答任何时间问题(现在几点/几点了/纽约几点/东京时间)前必须先调用本工具拿真实时间,严禁凭记忆或猜测回答——那样会给用户错误时间。以本工具返回的时间为准,忽略历史对话或长期记忆里提到过的任何时间(那些可能是之前错误巩固的幻觉)。拿到基准时间后由你换算成用户问的时区再回答。本工具返回服务器东八区(Asia/Shanghai)当前时间。用自然口语只给日期/星期/几点几分,不要附带「北京时间」「美东夏令时」「UTC±x」等时区术语。"""
    now = datetime.now(SERVER_ZONE)
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return (
        f"服务器当前时间(Asia/Shanghai, UTC{now.strftime('%z')}): "
        f"{now.strftime('%Y-%m-%d')} {weekdays[now.weekday()]} {now.strftime('%H:%M:%S')}\n"
        f"Unix时间戳: {int(now.timestamp())}"
    )


# ---------- 安全 AST 求值(无 eval) ----------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除、括号、幂运算、取模。输入一个数学表达式字符串,返回计算结果。"""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except (ValueError, SyntaxError, ZeroDivisionError, TypeError) as exc:
        return f"计算失败: {exc}"
