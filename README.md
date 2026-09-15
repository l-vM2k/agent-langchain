# langchain-ayaka

一个基于 Python LangChain + LangGraph 的陪伴型智能体系统。
个人学习项目,不商用。规格见 `spec/SPEC.md`。

## 架构

```
START -> prepare_soul -> retrieve_reme -> react_agent -> record_reme -> finalize_soul -> END
```

| 节点 | 职责 |
|------|------|
| prepare_soul | 人格加载 + 情绪感知 + 行为信号 + 记忆召回 + Prompt 组装 |
| retrieve_reme | 长期记忆检索注入 |
| react_agent | LLM ↔ 工具循环 |
| record_reme | 本轮对话写入长期记忆 |
| finalize_soul | 状态机转移 + 记忆抽取摄取 |

## 快速开始

```bash
# 安装
pip install -e ".[test]"

# 离线模式(不需要 API Key,用 FakeLLM 验证链路)
# Windows PowerShell
$env:AYAKA_FAKE_LLM = "1"; python scripts/run.py

# Linux/macOS
AYAKA_FAKE_LLM=1 python scripts/run.py

# 真实模式:设置环境变量后启动
$env:DASHSCOPE_API_KEY = "sk-xxx"; python scripts/run.py
```

## 测试

```bash
# 单元 + 端到端测试(30 个,全部离线可跑)
$env:AYAKA_FAKE_LLM = "1"; python -m pytest tests/ -q

# SSE 流式
curl -N -X POST http://127.0.0.1:8000/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"device_sn": "default", "message": "你好"}'

# 同步
curl -X POST http://127.0.0.1:8000/agent/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"device_sn": "default", "message": "你好"}'
```

## 配置

- `config/models/*.yaml` — LLM 定义(provider/api_host/model_name)
- `config/agents/*.yaml` — 智能体(sys_prompt/persona_type/soul_enabled/reme_enabled)
- `config/personality/strategies.yaml` — 3 底座行为权重/状态机参数/触发阈值
- `config/personality/prompt_rules.yaml` — 档位文案(9 维 × 4 档)
- `config/intent.yaml` — 意图快捷通道规则
- `config/tools/*.yaml` — REST/MCP 工具定义
- `config/knowledge/base.json` — 知识库条目

API Key 通过环境变量注入(`${DASHSCOPE_API_KEY}` 占位符),不写进文件。

## 数据

运行时数据在 `.data/`(gitignore):
- `soul_state.json` — 灵魂状态(PAD/trust/attachment,按 device×contact)
- `soul_memory.json` — 灵魂长期记忆(抽取的偏好/事件/边界)
- `reme/<device>_<contact>.json` — ReMe 对话记忆(按 workspace 隔离)

## 阶段进度

- [x] 阶段 1: 骨架 + LLM 工厂 + LangGraph ReAct + SSE 流式
- [x] 阶段 2: 工具系统(builtin/REST/knowledge/MCP 可选)
- [x] 阶段 3: 人格系统(14 维 + 3 底座 + 情绪 + 状态机 + Prompt 组装)
- [x] 阶段 4: 长期记忆(ReMe workspace 隔离 + 灵魂记忆抽取)
- [x] 阶段 5: 灵魂收尾(状态机转移) + 意图快捷通道
