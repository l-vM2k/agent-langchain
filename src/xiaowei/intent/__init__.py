"""Intent shortcut channel — mirrors intent package (IntentRule + IntentShortcutEnhancer).

When the user message matches a high-priority regex (asking the time, a
simple calc), we pre-invoke the tool and inject the result into the message,
bypassing the ReAct decision for a faster, cheaper reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class IntentRule:
    """单条意图识别规则 — IntentRule.java

    priority 数值越小优先级越高(升序匹配)。
    enhance_template 支持 {message} / {tool_result} 占位符。
    """

    intent_type: str
    pattern: str
    priority: int = 100
    enhance_template: str = ""


@dataclass
class IntentRuleSet:
    """意图快捷通道配置 — IntentRuleProperties.java"""

    enabled: bool = True
    rules: list[IntentRule] = field(default_factory=list)

    def sorted_rules(self) -> list[IntentRule]:
        return sorted(self.rules, key=lambda r: r.priority)


class IntentClassifier:
    """意图分类器 — RegexIntentClassifier.java"""

    def __init__(self, rule_set: IntentRuleSet) -> None:
        self._rule_set = rule_set
        self._compiled: list[tuple[re.Pattern[str], IntentRule]] = []
        for rule in rule_set.rules:
            try:
                self._compiled.append((re.compile(rule.pattern), rule))
            except re.error:
                continue

    def classify(self, message: str) -> IntentRule | None:
        """返回第一条命中的规则(按 priority 升序)。"""
        if not self._rule_set.enabled or not message:
            return None
        for pattern, rule in self._compiled:
            if pattern.search(message):
                return rule
        return None


class IntentShortcutToolRegistry:
    """意图 → 工具映射 — IntentShortcutToolRegistry.java

    GET_TIME/GET_DATE/GET_WEEKDAY → current_time
    SIMPLE_CALC → calculator
    """

    def __init__(self) -> None:
        self._handlers: dict[str, callable] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        from xiaowei.tools.builtin import calculator, current_time

        for intent in ("GET_TIME", "GET_DATE", "GET_WEEKDAY"):
            self._handlers[intent] = lambda msg: current_time.invoke({})
        self._handlers["SIMPLE_CALC"] = lambda msg: _calc_from_message(calculator, msg)

    def invoke(self, intent_type: str, message: str) -> str | None:
        """调用意图对应的工具,返回结果字符串;无注册/异常返回 None。"""
        handler = self._handlers.get(intent_type)
        if handler is None:
            return None
        try:
            return handler(message)
        except Exception:  # noqa: BLE001 — 快捷通道失败降级原消息
            return None


def _calc_from_message(calculator_tool, message: str) -> str | None:
    """从消息中提取 a op b 算式并求值 — CalculatorTool.calculateFromMessage

    边界安全: 匹配到的 a op b 不能是更大表达式的一部分
    (负数前导符/链式/科学计数法一律降级返回 None)。
    """
    import re

    expr = re.compile(r"(\d+(?:\.\d+)?)\s*([+\-*/×÷])\s*(\d+(?:\.\d+)?)")
    m = expr.search(message or "")
    if not m:
        return None
    if not _is_boundary_safe(message, m.start(), m.end()):
        return None
    a = float(m.group(1))
    b = float(m.group(3))
    op = m.group(2).replace("×", "*").replace("÷", "/")
    try:
        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        elif op == "*":
            result = a * b
        else:
            if b == 0:
                return None
            result = a / b
    except (ValueError, ZeroDivisionError):
        return None
    if result == int(result):
        return str(int(result))
    return str(result)


def _is_boundary_safe(message: str, start: int, end: int) -> bool:
    """校验匹配区间前后边界 — CalculatorTool.isBoundarySafe"""
    before = message[:start].strip()
    after = message[end:].strip()

    def is_token_char(c: str) -> bool:
        return c.isdigit() or c in ".Ee" or c in "+-*/×÷"

    def is_operator_char(c: str) -> bool:
        return c in "+-*/×÷"

    if before and is_token_char(before[-1]):
        return False
    if after and is_operator_char(after[0]):
        return False
    return True


class IntentShortcutEnhancer:
    """意图快捷消息增强器 — IntentShortcutEnhancer.java

    命中 → 调工具 → 渲染 enhance_template → 返回增强消息;
    未命中/异常/空结果 → 原样返回(零回归)。
    """

    def __init__(self, classifier: IntentClassifier, registry: IntentShortcutToolRegistry) -> None:
        self._classifier = classifier
        self._registry = registry

    def enhance(self, message: str) -> str:
        """增强用户消息,永远返回非 None(最坏原样)。"""
        if not message or not message.strip():
            return message
        try:
            rule = self._classifier.classify(message)
            if rule is None:
                return message
            tool_result = self._registry.invoke(rule.intent_type, message)
            if not tool_result or not str(tool_result).strip():
                return message
            if not rule.enhance_template:
                return message
            return rule.enhance_template.format(message=message, tool_result=tool_result)
        except Exception:  # noqa: BLE001 — 快捷通道失败降级原消息
            return message
