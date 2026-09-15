"""Tests for tools, memory (ReMe), intent shortcut, and the full graph."""

from __future__ import annotations

import asyncio

from xiaowei.intent import (
    IntentClassifier,
    IntentRule,
    IntentRuleSet,
    IntentShortcutEnhancer,
    IntentShortcutToolRegistry,
)
from xiaowei.memory.anonymous import derive_anonymous_contact_id, is_anonymous
from xiaowei.memory.reme_store import ReMeMemoryStore
from xiaowei.tools.builtin import calculator, current_time


# ---------- builtin tools ----------

def test_current_time_tool():
    result = current_time.invoke({})
    assert "服务器当前时间" in result
    assert "Asia/Shanghai" in result


def test_calculator_tool():
    assert calculator.invoke({"expression": "3+5"}) == "8"
    assert calculator.invoke({"expression": "(2+3)*4"}) == "20"
    assert calculator.invoke({"expression": "2**10"}) == "1024"
    # 非法输入返回错误信息而非抛异常
    assert "计算失败" in calculator.invoke({"expression": "__import__('os')"})


# ---------- anonymous contact id ----------

def test_anonymous_id_stable():
    a = derive_anonymous_contact_id("dev1", "sess1")
    b = derive_anonymous_contact_id("dev1", "sess1")
    c = derive_anonymous_contact_id("dev1", "sess2")
    assert a == b
    assert a != c
    assert is_anonymous(a)
    assert not is_anonymous("real-contact-1")


# ---------- ReMe memory ----------

def test_reme_roundtrip(tmp_path):
    store = ReMeMemoryStore(tmp_path / "reme")

    async def _run():
        # 写入两轮
        await store.record("d1", "c1", "我叫张三,在杭州工作", "你好张三!")
        await store.record("d1", "c1", "今天天气不错", "是呀,适合出门")
        # 检索
        return await store.retrieve("d1", "c1", "张三 杭州")

    result = asyncio.run(_run())
    assert result is not None
    assert "张三" in result
    assert "<long_term_memory>" in result


def test_reme_anonymous_skip(tmp_path):
    store = ReMeMemoryStore(tmp_path / "reme")

    async def _run():
        anon = derive_anonymous_contact_id("d1", "s1")
        await store.record("d1", anon, "我叫张三", "你好")
        return await store.retrieve("d1", anon, "张三")

    result = asyncio.run(_run())
    assert result is None  # 匿名跳过存取


def test_reme_isolation(tmp_path):
    store = ReMeMemoryStore(tmp_path / "reme")

    async def _run():
        await store.record("d1", "c1", "用户A的秘密", "ok")
        await store.record("d1", "c2", "用户B的秘密", "ok")
        return (
            await store.retrieve("d1", "c1", "秘密"),
            await store.retrieve("d1", "c2", "秘密"),
        )

    a, b = asyncio.run(_run())
    assert a is not None and "用户A" in a
    assert b is not None and "用户B" in b
    assert "用户B" not in a  # workspace 隔离


# ---------- intent shortcut ----------

def _rule_set() -> IntentRuleSet:
    return IntentRuleSet(
        enabled=True,
        rules=[
            IntentRule(
                intent_type="GET_TIME",
                pattern=r"(?i).*(现在几点|几点了).*",
                priority=10,
                enhance_template="用户问:{message}\n[系统已代为查询:{tool_result}。请直接简短回答。]",
            ),
            IntentRule(
                intent_type="SIMPLE_CALC",
                pattern=r".*[0-9.]+\s*[+\-*/×÷]\s*[0-9.]+.*",
                priority=40,
                enhance_template="用户问:{message}\n[系统已代为计算:结果 = {tool_result}。请直接简短回答。]",
            ),
        ],
    )


def test_intent_classify_time():
    classifier = IntentClassifier(_rule_set())
    rule = classifier.classify("现在几点了?")
    assert rule is not None
    assert rule.intent_type == "GET_TIME"


def test_intent_enhance_time():
    enhancer = IntentShortcutEnhancer(IntentClassifier(_rule_set()), IntentShortcutToolRegistry())
    enhanced = enhancer.enhance("现在几点了?")
    assert "系统已代为查询" in enhanced
    assert "现在几点了?" in enhanced


def test_intent_enhance_calc():
    enhancer = IntentShortcutEnhancer(IntentClassifier(_rule_set()), IntentShortcutToolRegistry())
    enhanced = enhancer.enhance("3+5等于多少")
    assert "结果 = 8" in enhanced


def test_intent_enhance_no_match():
    enhancer = IntentShortcutEnhancer(IntentClassifier(_rule_set()), IntentShortcutToolRegistry())
    original = "今天心情不太好"
    assert enhancer.enhance(original) == original


def test_intent_calc_boundary_negative():
    # "-3+5" 里的 "3+5" 是负数前导符的一部分 → 降级原消息
    enhancer = IntentShortcutEnhancer(IntentClassifier(_rule_set()), IntentShortcutToolRegistry())
    original = "温度是-3+5度"
    # pattern 命中(含 3+5),但边界不安全 → 工具返回 None → 原样返回
    assert enhancer.enhance(original) == original


# ---------- full graph (fake LLM) ----------

def test_full_graph_end_to_end(tmp_path):
    import os
    os.environ["XIAOWEI_FAKE_LLM"] = "1"

    from xiaowei.agent.manager import AgentManager
    from xiaowei.config import ConfigLoader
    from pathlib import Path

    config_dir = Path(__file__).resolve().parents[1] / "config"
    loader = ConfigLoader(config_dir)
    mgr = AgentManager(loader, tmp_path / "data")
    agent = mgr.get_agent("default")

    config = {"configurable": {"thread_id": "test-session-1"}}

    async def _run():
        result = await agent.ainvoke(
            {
                "messages": [("user", "你好,我叫张三")],
                "device_sn": "default",
                "contact_id": "contact-1",
                "session_id": "test-session-1",
            },
            config=config,
        )
        return result

    result = asyncio.run(_run())
    messages = result["messages"]
    # 最后一条是 assistant 回复
    last = messages[-1]
    assert getattr(last, "type", "") == "ai" or last.__class__.__name__ == "AIMessage"
    # soul_context 已填充
    assert result.get("soul_context") is not None
    # 灵魂 prompt 已组装
    soul_ctx = result["soul_context"]
    assert len(soul_ctx.prompt) > 0
    assert "本轮表达倾向" in soul_ctx.prompt


def test_full_graph_session_isolation(tmp_path):
    import os
    os.environ["XIAOWEI_FAKE_LLM"] = "1"

    from xiaowei.agent.manager import AgentManager
    from xiaowei.config import ConfigLoader
    from pathlib import Path

    config_dir = Path(__file__).resolve().parents[1] / "config"
    loader = ConfigLoader(config_dir)
    mgr = AgentManager(loader, tmp_path / "data")
    agent = mgr.get_agent("default")

    async def _run():
        cfg_a = {"configurable": {"thread_id": "sess-a"}}
        cfg_b = {"configurable": {"thread_id": "sess-b"}}
        base = {
            "device_sn": "default",
            "contact_id": "contact-1",
        }
        r1 = await agent.ainvoke(
            {**base, "session_id": "sess-a", "messages": [("user", "你好")]},
            config=cfg_a,
        )
        r2 = await agent.ainvoke(
            {**base, "session_id": "sess-b", "messages": [("user", "你好")]},
            config=cfg_b,
        )
        return r1, r2

    r1, r2 = asyncio.run(_run())
    # 两个 thread 各自独立: r1 只有 2 条(human+ai),r2 可能多一条 ReMe 注入的 tool message
    # 关键是 r1 不包含 r2 的消息(隔离),且各自都产生了 ai 回复
    assert len(r1["messages"]) >= 2
    assert len(r2["messages"]) >= 2
    # r1 的消息里不应有 r2 的痕迹(内容一致无法区分,但数量级应接近)
    assert abs(len(r1["messages"]) - len(r2["messages"])) <= 1
