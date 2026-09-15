# langchain-ayaka 规格说明书

> 一个基于 Python LangChain + LangGraph 的陪伴型智能体系统。
> 本文件是**唯一权威规格**,后续编码严格按此执行。

---

## 0. 项目定位与目标

### 0.1 为什么做这个项目

通过实现一个**真实生产级智能体系统**,
在实战中掌握大模型应用开发的核心能力。

**本项目聚焦对话闭环**,不做场景引擎、不做主动关怀、不做设备接入。

### 0.2 学什么

| 能力 | LangChain/LangGraph 对应 |
|------|--------------------------|
| LLM 抽象与多 provider 路由 | `ChatOpenAI` / `ChatOllama` |
| ReAct Agent 与工具调用 | `create_react_agent` / 自定义 `StateGraph` |
| MCP 协议集成 | `langchain-mcp-adapters` |
| 流式输出 SSE | `astream_events` + FastAPI `StreamingResponse` |
| 长期记忆与向量检索 | LangGraph checkpoint + 自定义 store |
| 人格系统与 Prompt 工程 | LangGraph 节点 + `ChatPromptTemplate` |
| 灵魂状态机 | LangGraph 节点 + 纯函数 |
| 多用户/多会话隔离 | LangGraph `thread_id` + checkpoint |

### 0.3 不做什么

- **不做设备接入**: 不接 MQTT/机器人硬件
- **不做场景引擎**: 不做 scenario 包
- **不做主动关怀**: 不做 proactive 包
- **不做前端**: 只做后端 API + SSE,用 curl 测试

### 0.4 技术栈

```
Python 3.11+
langchain >= 0.3
langgraph >= 0.2
langchain-mcp-adapters  # MCP 工具(可选)
langchain-openai        # OpenAI/DashScope/DeepSeek 等
langchain-ollama        # Ollama(本地模型)
langchain-community     # FAISS 向量存储
fastapi                 # API + SSE
uvicorn                 # ASGI server
pydantic                # 数据模型
pyyaml                  # 配置
httpx                   # REST 工具
```

### 0.5 部署目标

2 核 2G 服务器,个人使用,不商用.
LLM 走远程 API,不在本地跑模型.
记忆用本地 SQLite/JSON,不跑独立服务.

---

## 1. 系统架构

### 1.1 本项目涉及的层

```
┌─────────────────────────────────────────────────────────┐
│ 1. Agent 生命周期层                                      │
│    AgentManager (懒加载/热重载) + ConfigResolver          │
├─────────────────────────────────────────────────────────┤
│ 2. 对话入口层                                            │
│    ChatService (SSE/Sync)                                │
├─────────────────────────────────────────────────────────┤
│ 3. 处理链                                                │
│    SoulPrompt → Context → ModelOptions                   │
│    → ReMeMemory → AsyncCompaction → PromptTracer        │
├─────────────────────────────────────────────────────────┤
│ 4. 灵魂系统 (Soul) — 人格/情绪/状态机/记忆               │
│    SoulOrchestrator → Preparer / Finalizer               │
├─────────────────────────────────────────────────────────┤
│ 5. 长期记忆 (ReMe)                                       │
│    ReMeMemoryStore (deviceSn:contactId 隔离)             │
│    ReMeMemoryMiddleware (检索注入 + 异步写入)            │
└─────────────────────────────────────────────────────────┘
```

不做的层: 场景引擎(6)、主动关怀(7)、设备接入相关.

### 1.2 一次对话的完整流程(本项目范围)

```
用户消息 → ChatService.chatStream
  ├─ 解析 deviceSn / contactId / sessionId
  ├─ AgentManager.getAgent(deviceSn)  ← 懒加载 Agent
  ├─ SoulOrchestrator.prepare()       ← 灵魂上下文准备
  │   ├─ 加载人格配置 (PersonalityProfile: 14 维 + 3 底座)
  │   ├─ 加载灵魂状态 (SoulState: PAD + trust/attachment)
  │   ├─ 情绪感知 (EmotionAnalyzer: 关键词 + 否定前缀)
  │   ├─ 行为信号计算 (BehaviorSignalGenerator: 6 维加权)
  │   ├─ 记忆召回 (SoulMemorySelector: 相关性选择)
  │   └─ Prompt 组装 (SoulPromptComposer: 拼装人格/情绪/记忆段)
  ├─ agent.streamEvents(messages, ctx)
  │   └─ 处理链执行:
  │       1. SoulSystemPromptMiddleware  → 注入人格 prompt
  │       2. ContextSystemPromptMiddleware → 注入前端上下文
  │       3. ReMeMemoryMiddleware → 检索长期记忆注入
  │       → ReAct 循环 (LLM ↔ Tool)
  ├─ 事件流 → SSE 映射 (token/thinking/tool_call/done)
  └─ SoulOrchestrator.finalize()  ← 灵魂收尾
      ├─ 状态机转移 (SoulStateMachine.next)
      └─ 记忆摄取/画像成长
```

---

## 2. 架构设计

### 2.1 核心设计: 处理链 → LangGraph StateGraph

系统的处理链本质是 **"预处理 → ReAct 循环 → 后处理"**。
LangGraph 的 `StateGraph` 天然表达这个结构:

```
          ┌──────────────┐
          │ prepare_soul  │ ← 人格/情绪/状态/记忆召回/Prompt 组装
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │ retrieve_reme │ ← 长期记忆检索注入
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │  react_agent  │ ← LLM ↔ Tool 循环
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │  record_reme  │ ← 长期记忆写入
          └──────┬───────┘
                 ▼
          ┌──────────────┐
          │ finalize_soul │ ← 状态机转移
          └──────┬───────┘
                 ▼
              END
```

### 2.2 项目目录结构

```
langchain-ayaka/
├── spec/
│   └── SPEC.md                 # 本文件
├── pyproject.toml
├── README.md
├── config/                      # YAML 配置
│   ├── agents/
│   │   └── default.yaml         # 智能体配置
│   ├── models/
│   │   ├── qwen.yaml            # 通义千问
│   │   └── deepseek.yaml        # DeepSeek
│   ├── tools/
│   │   ├── mcp_local.yaml       # 本地 MCP 工具
│   │   └── mcp_remote.yaml      # 远程 MCP 工具
│   └── personality/
│       ├── dimensions.yaml      # 14 维度默认值
│       ├── strategies.yaml      # 3 底座策略(权重/deltas/系数)
│       └── prompt_rules.yaml    # 档位文案
├── src/
│   └── ayaka/
│       ├── __init__.py
│       ├── config.py            # 配置加载
│       ├── llm/
│       │   ├── __init__.py
│       │   └── factory.py        # LLM 工厂
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── builtin.py        # 内置工具(current_time/calculator)
│       │   ├── mcp.py            # MCP 工具
│       │   ├── rest.py           # REST 工具
│       │   └── knowledge.py      # 知识库检索
│       ├── soul/                # 灵魂系统
│       │   ├── __init__.py
│       │   ├── models.py         # SoulContext/SoulState/PersonalityProfile
│       │   ├── emotion.py        # 情绪感知
│       │   ├── behavior.py       # 行为信号计算
│       │   ├── state_machine.py  # 状态机
│       │   ├── prompt_composer.py
│       │   ├── memory_selector.py
│       │   └── orchestrator.py   # Preparer / Finalizer
│       ├── memory/               # 长期记忆
│       │   ├── __init__.py
│       │   ├── reme_store.py     # ReMe 存储
│       │   └── anonymous.py      # 匿名 ID
│       ├── agent/                # Agent 核心
│       │   ├── __init__.py
│       │   ├── manager.py        # Agent 生命周期
│       │   └── graph.py           # LangGraph StateGraph 定义
│       ├── api/                  # API 层
│       │   ├── __init__.py
│       │   ├── server.py         # FastAPI app
│       │   └── sse.py            # SSE 事件映射
│       └── intent/               # 意图快捷
│           ├── __init__.py
│           └── enhancer.py
├── tests/
│   ├── test_llm.py
│   ├── test_tools.py
│   ├── test_soul.py
│   └── test_agent.py
└── scripts/
    └── run.py                    # 启动脚本
```

---

## 3. 模块规格

### 3.1 LLM 工厂 (`llm/factory.py`)

**路由行为**:
- 按 `providerCode` 路由: `qianwen` → DashScope, `ollama` → Ollama, 其他 → OpenAI 兼容
- 参数: `apiHost`, `apiKey`, `modelName`, `stream`

**Python 实现**:

```python
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def create_llm(model_config: ModelConfig) -> BaseChatModel:
    match model_config.provider_code:
        case "ollama":
            return ChatOllama(
                base_url=model_config.api_host,
                model=model_config.model_name,
            )
        case _:  # qianwen/deepseek/openai/custom_api 等
            return ChatOpenAI(
                base_url=model_config.api_host,
                api_key=model_config.api_key,
                model=model_config.model_name,
                streaming=model_config.stream,
            )
```

**配置格式** (`config/models/qwen.yaml`):
```yaml
model_id: 1
name: "qwen-plus"
provider_code: "qianwen"
api_host: "https://dashscope.aliyuncs.com/compatible-mode/v1"
api_key: "${DASHSCOPE_API_KEY}"
model_name: "qwen-plus"
stream: true
model_param_json:
  enable_thinking: false
  temperature: 0.7
```

**学习要点**:
- LangChain 的 `BaseChatModel` 抽象
- `ChatOpenAI` 兼容所有 OpenAI 协议的模型(DeepSeek/DashScope 等)
- `streaming=True` 开启流式

---

### 3.2 工具系统 (`tools/`)

**支持 4 种工具类型**(不做 RPC/设备工具):

| 类型 | 本项目 | 说明 |
|------|--------|------|
| LOCAL | ✅ | stdio MCP Server,`langchain-mcp-adapters` |
| REMOTE | ✅ | streamableHttp/SSE MCP,`langchain-mcp-adapters` |
| BUILTIN | ✅ | `@tool` 装饰器 |
| REST | ✅ | `StructuredTool` + httpx |

#### 3.2.1 内置工具 (`tools/builtin.py`)

```python
from langchain_core.tools import tool

@tool
def current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前时间。按用户询问的时区返回。"""
    from datetime import datetime
    import zoneinfo
    tz = zoneinfo.ZoneInfo(timezone)
    return datetime.now(tz).strftime("%Y年%m月%d日 %H:%M:%S %Z")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除、括号、幂运算。"""
    import ast, operator
    # ... 安全的 AST 求值
```

#### 3.2.2 MCP 工具 (`tools/mcp.py`)

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async def create_mcp_tools(tool_configs: list[ToolConfig]) -> list:
    servers = {}
    for cfg in tool_configs:
        if cfg.type == "LOCAL":
            servers[cfg.name] = {
                "command": cfg.config["command"],
                "args": cfg.config.get("args", []),
                "env": cfg.config.get("env", {}),
                "transport": "stdio",
            }
        elif cfg.type == "REMOTE":
            servers[cfg.name] = {
                "url": cfg.config["baseUrl"],
                "transport": cfg.config.get("transport", "streamable_http"),
            }
    client = MultiServerMCPClient(servers)
    return await client.get_tools()
```

#### 3.2.3 REST 工具 (`tools/rest.py`)

```python
from langchain_core.tools import StructuredTool
import httpx

def create_rest_tool(name: str, description: str, params: dict) -> StructuredTool:
    async def _call(**kwargs) -> str:
        async with httpx.AsyncClient(timeout=params.get("timeout", 10)) as client:
            resp = await client.request(
                params.get("method", "POST"), params["url"],
                headers=params.get("headers", {}), json=kwargs,
            )
            return resp.text
    return StructuredTool.from_function(
        coroutine=_call, name=name, description=description,
    )
```

#### 3.2.4 知识库检索 (`tools/knowledge.py`)

```python
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

@tool
async def knowledge_search(query: str) -> str:
    """在知识库中检索相关信息。"""
    vectorstore = FAISS.load_local("config/knowledge/index", OpenAIEmbeddings())
    docs = vectorstore.similarity_search(query, k=3)
    return "\n---\n".join(d.page_content for d in docs)
```

**学习要点**:
- `@tool` 装饰器 vs `StructuredTool`
- MCP 协议集成
- 工具的 `args_schema` (Pydantic)

---

### 3.3 灵魂系统 (`soul/`)

**核心模块**(最值得学的部分)

#### 3.3.1 数据模型 (`soul/models.py`)

```python
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional

class PersonaType(str, Enum):
    WARM_COMPANION = "WARM_COMPANION"
    RATIONAL_ASSISTANT = "RATIONAL_ASSISTANT"
    QUIET_GUARDIAN = "QUIET_GUARDIAN"

class EmotionType(str, Enum):
    POSITIVE = "POSITIVE"
    SAD = "SAD"
    ANXIOUS = "ANXIOUS"
    ANGRY = "ANGRY"
    GRATEFUL = "GRATEFUL"
    NEGATIVE_FEEDBACK = "NEGATIVE_FEEDBACK"
    CONFIDING = "CONFIDING"
    NEUTRAL = "NEUTRAL"

class RelationshipTone(str, Enum):
    STRANGER = "STRANGER"
    FAMILIAR = "FAMILIAR"
    TRUSTED = "TRUSTED"
    INTIMATE = "INTIMATE"
    FAMILY_COMPANION = "FAMILY_COMPANION"

class PersonalityDimensions(BaseModel):
    """人格 14 维度,取值 [0,1]"""
    empathy: float = 0.55
    dominance: float = 0.45
    formality: float = 0.30
    curiosity: float = 0.50
    relationship_distance: float = 0.45
    humor: float = 0.40
    drive_connection: float = 0.55
    drive_novelty: float = 0.50
    drive_expression: float = 0.50
    drive_safety: float = 0.70
    drive_play: float = 0.35
    baseline_valence: float = 0.15
    baseline_arousal: float = 0.0
    baseline_dominance: float = 0.0

class PersonalityProfile(BaseModel):
    device_sn: str
    enabled: bool = True
    persona_type: PersonaType = PersonaType.WARM_COMPANION
    dimensions: PersonalityDimensions = PersonalityDimensions()
    sys_prompt: str = ""

class SoulState(BaseModel):
    """会话维度动态灵魂状态"""
    device_sn: str
    contact_id: str
    session_id: str
    valence: float = 0.15      # PAD: 效价 [-1,1]
    arousal: float = 0.0       # PAD: 唤醒 [-1,1]
    dominance: float = 0.0    # PAD: 主导 [-1,1]
    frustration: float = 0.0   # [0,1]
    attachment: float = 0.1   # [0,1]
    trust: float = 0.5        # [0,1]
    boredom: float = 0.0      # [0,1]
    turn_count: int = 0
    last_user_emotion: EmotionType = EmotionType.NEUTRAL
    last_summary: Optional[str] = None
    last_interaction_at: datetime = Field(default_factory=datetime.now)

class BehaviorSignal(BaseModel):
    """行为信号 6 维,[0,1]"""
    warmth: float = 0.5
    initiative: float = 0.45
    curiosity: float = 0.5
    restraint: float = 0.6
    playfulness: float = 0.4
    action_desire: float = 0.35
    relationship_tone: RelationshipTone = RelationshipTone.STRANGER

class SoulPerception(BaseModel):
    """本轮感知结果"""
    emotion_type: EmotionType = EmotionType.NEUTRAL
    valence_delta: float = 0.0
    arousal_delta: float = 0.0
    dominance_delta: float = 0.0
    trust_delta: float = 0.0
    attachment_delta: float = 0.0
    frustration_delta: float = 0.0
    confiding: bool = False
    care_needed: bool = False
    restraint_hint: bool = False

class SoulContext(BaseModel):
    """灵魂上下文 — 本轮对话的完整上下文"""
    device_sn: str
    contact_id: str
    session_id: str
    personality_config: PersonalityProfile
    soul_state: SoulState
    perception: SoulPerception
    behavior_signal: BehaviorSignal
    relationship_tone: RelationshipTone
    care_success_score: float = 0.5
    selected_memories: list = []
    prompt: str = ""
    enabled: bool = True
```

#### 3.3.2 情绪感知 (`soul/emotion.py`)

**行为**:
1. 关键词命中优先级: ANGRY > NEGATIVE_FEEDBACK > GRATEFUL > SAD > ANXIOUS > POSITIVE > CONFIDING(long) > NEUTRAL
2. 否定前缀检测: 关键词前 1-2 字为 `不/没/别/未/有没` 则该位置不命中
3. 每种情绪有 6 维 deltas (valence/arousal/dominance/trust/attachment/frustration)

```python
class EmotionAnalyzer:
    """轻量规则情绪识别器(关键词 + 否定前缀)"""

    NEGATION_PREFIXES_1 = {"不", "没", "别", "未"}
    NEGATION_PREFIXES_2 = {"有没"}

    EMOTION_LEXICONS = {
        EmotionType.GRATEFUL: ["谢谢", "感谢", "辛苦", "帮了我"],
        EmotionType.ANGRY: ["生气", "气死", "垃圾", "烦", "太差"],
        EmotionType.NEGATIVE_FEEDBACK: ["不对", "错了", "没用", "不行", "失败"],
        EmotionType.SAD: ["难过", "伤心", "委屈", "失落", "哭", "累", "疲惫",
                         "撑不住", "心累", "心里空", "空虚", "孤独", "一个人",
                         "没意思", "抑郁", "想哭", "绝望", "emo"],
        EmotionType.ANXIOUS: ["焦虑", "担心", "害怕", "压力", "怎么办",
                             "失眠", "睡不着", "不安", "紧张", "心慌",
                             "喘不过气", "崩溃"],
        EmotionType.POSITIVE: ["开心", "喜欢", "太好了", "棒", "哈哈", "不错"],
    }

    EMOTION_DELTAS = {
        EmotionType.ANGRY: {"valence": -0.35, "arousal": 0.25, "dominance": -0.15,
                           "trust": -0.08, "attachment": 0.0, "frustration": 0.18},
        EmotionType.NEGATIVE_FEEDBACK: {"valence": -0.25, "arousal": 0.10,
                                        "dominance": -0.12, "trust": -0.06,
                                        "attachment": 0.0, "frustration": 0.14},
        EmotionType.GRATEFUL: {"valence": 0.24, "arousal": 0.05, "dominance": 0.10,
                              "trust": 0.08, "attachment": 0.03, "frustration": -0.03},
        EmotionType.SAD: {"valence": -0.32, "arousal": -0.08, "dominance": 0.0,
                        "trust": 0.0, "attachment": 0.06, "frustration": 0.02},
        EmotionType.ANXIOUS: {"valence": -0.24, "arousal": 0.20, "dominance": 0.0,
                            "trust": 0.0, "attachment": 0.04, "frustration": 0.05},
        EmotionType.POSITIVE: {"valence": 0.26, "arousal": 0.12, "dominance": 0.08,
                              "trust": 0.03, "attachment": 0.02, "frustration": -0.02},
        EmotionType.CONFIDING: {"valence": -0.08, "arousal": -0.02, "dominance": 0.0,
                              "trust": 0.01, "attachment": 0.05, "frustration": 0.0},
        EmotionType.NEUTRAL: {},
    }

    def analyze(self, text: str) -> SoulPerception:
        """分析文本情绪,返回感知结果。"""
        # 1. 关键词匹配(按优先级)
        # 2. 否定前缀检测
        # 3. 组装 6 维 deltas
        # 4. care_needed = SAD/ANXIOUS/CONFIDING
        # 5. restraint_hint = ANGRY/NEGATIVE_FEEDBACK
```

#### 3.3.3 行为信号计算 (`soul/behavior.py`)

**行为**: 6 个行为信号加权计算,权重从 `BehaviorStrategy` 读取。

```python
class BehaviorSignalGenerator:
    """将人格维度 + 灵魂状态 + 情绪 → 6 维行为信号"""

    BEHAVIOR_WEIGHTS = {
        "warmth": {"empathy": 0.30, "attachment": 0.25, "trust": 0.20, "driveConnection": 0.25},
        "initiative": {"dominance": 0.35, "driveConnection": 0.25, "boredom": 0.15, "inverseDriveSafety": 0.25},
        "curiosity": {"curiosity": 0.60, "driveNovelty": 0.40},
        "restraint": {"driveSafety": 0.45, "formality": 0.10, "frustration": 0.35, "inverseTrust": 0.10},
        "playfulness": {"humor": 0.25, "drivePlay": 0.45, "valenceNormalized": 0.30},
        "actionDesire": {"driveExpression": 0.50, "arousalNormalized": 0.25, "playfulness": 0.20, "driveSafety": 0.10},
    }

    def generate(self, profile: PersonalityProfile, state: SoulState,
                 emotion: EmotionType) -> BehaviorSignal:
        # 1. 按 BEHAVIOR_WEIGHTS 加权计算 6 个信号
        # 2. 情绪修正(ANGRY/NEGATIVE_FEEDBACK: initiative-0.20, restraint+0.08)
        # 3. clamp[0,1]
        # 4. 关系档位推断
```

**权重矩阵详见附录 B**。

#### 3.3.4 状态机 (`soul/state_machine.py`)

**行为**:
- 衰减率: `clamp(0.05 + minutes/240 * 0.30, 0.05, 0.35)`
- PAD 衰减回 baseline + 叠加信号 delta
- 工具成功: trust +0.03, frustration -0.05
- 工具失败: trust -0.03, frustration +0.15
- 短回复无工具: boredom +0.03

```python
class SoulStateMachine:
    def next(self, current: SoulState, perception: SoulPerception,
             trace: SoulTurnTrace, now: datetime) -> SoulState:
        decay = self._decay_rate(current.last_interaction_at, now)
        valence = self._decay_toward(current.valence, 0.15, decay) + perception.valence_delta
        arousal = self._decay_toward(current.arousal, 0, decay) + perception.arousal_delta
        dominance = self._decay_toward(current.dominance, 0, decay) + perception.dominance_delta
        frustration = self._decay_toward(current.frustration, 0, decay) + perception.frustration_delta
        attachment = current.attachment + perception.attachment_delta + 0.01  # 每轮自然递增
        trust = current.trust + perception.trust_delta

        if trace.tool_success_count > 0:
            trust += 0.03 * trace.tool_success_count
            frustration -= 0.05 * trace.tool_success_count
        if trace.tool_failure_count > 0:
            trust -= 0.03 * trace.tool_failure_count
            frustration += 0.15 * trace.tool_failure_count
        if trace.assistant_text_length < 8 and trace.tool_call_count == 0:
            boredom += 0.03

        return SoulState(
            ...,
            valence=clamp(valence, -1, 1),
            arousal=clamp(arousal, -1, 1),
            # ... clamp 各维度
            turn_count=current.turn_count + 1,
            last_user_emotion=perception.emotion_type,
        )
```

#### 3.3.5 Prompt 组装 (`soul/prompt_composer.py`)

**行为**: 拼装以下段落到系统提示:
1. 身份 ownership 声明("请自然地成为这个人")
2. 【本轮表达倾向】6 维行为信号按四档取文案
3. 【此刻情绪底色】PAD 三维按四档取文案
4. 行为要求(任务优先/工具调用/人格自然)
5. 关系档位文案
6. 【关于对方】联系人档案(可选)
7. 【长期记忆】召回的记忆条目

```python
class SoulPromptComposer:
    def compose(self, config, state, emotion, signal, tone,
                memories, contact=None) -> str:
        prompt = "下面描述的是你此刻真实的状态和心情。请自然地成为这个人...\n\n"
        prompt += self._signal_section(signal, rule_texts)    # 6 维行为信号
        prompt += self._pad_section(state, rule_texts)        # PAD 三维
        prompt += self._behavior_requirements()               # 行为要求
        prompt += self._relationship_tone(tone)               # 关系档位
        if contact:
            prompt += self._contact_section(contact)          # 联系人档案
        if memories:
            prompt += self._memory_section(memories)          # 长期记忆
        return self._fit_budget(prompt, max_chars=6000)
```

**档位文案** (`config/personality/prompt_rules.yaml`):
```yaml
rules:
  - dimension: "sig_warmth"
    band: "LOW"
    prompt_text: "陪伴倾向偏低,保持适度距离"
  - dimension: "sig_warmth"
    band: "MID_LOW"
    prompt_text: "温和友好,不过度亲密"
  # ... 每个维度 × 4 档
```

#### 3.3.6 灵魂编排器 (`soul/orchestrator.py`)

```python
class SoulOrchestrator:
    def __init__(self, ...):
        self.preparer = SoulTurnPreparer(...)
        self.finalizer = SoulTurnFinalizer(...)

    def prepare(self, device_sn, contact_id, session_id, user_text) -> SoulContext:
        return self.preparer.prepare(device_sn, contact_id, session_id, user_text)

    def finalize(self, context, user_text, assistant_text, trace):
        self.finalizer.finalize(context, user_text, assistant_text, trace)
```

**学习要点**:
- Pydantic 数据建模
- Enum 与类型安全
- 纯函数式状态机(无副作用计算)
- Prompt 工程(分段拼装 + 档位文案)

---

### 3.4 长期记忆 (`memory/`)

#### 3.4.1 ReMe 存储 (`memory/reme_store.py`)

**行为**:
- 按 `(deviceSn, contactId)` 隔离 workspace
- `workspace_id = deviceSn + "_" + contactId`
- 匿名对话跳过
- record 失败静默吞掉

**Python 实现** (阶段1: 本地 JSON; 阶段2: SQLite + FAISS 向量检索):

```python
class ReMeMemoryStore:
    """长期记忆存储,按 (device_sn, contact_id) 隔离"""

    def __init__(self, base_dir: str = ".data/reme"):
        self.base_dir = base_dir

    async def retrieve(self, device_sn: str, contact_id: str,
                       query: str) -> str | None:
        """检索长期记忆。"""
        if is_anonymous(contact_id):
            return None
        # 阶段1: 关键词匹配; 阶段2: FAISS 向量检索
        memories = self._load(device_sn, contact_id)
        # ... 相关性排序,返回 top-k

    async def record(self, device_sn: str, contact_id: str,
                     messages: list) -> None:
        """写入记忆。失败静默。"""
        if is_anonymous(contact_id):
            return
        # 过滤掉 knowledge_search / long_term_memory 的 TOOL 消息
        filtered = self._filter(messages)
        # ... 追加写入
```

#### 3.4.2 匿名 ID (`memory/anonymous.py`)

```python
import uuid

ANON_PREFIX = "anon-"

def derive_anonymous_contact_id(device_sn: str, session_id: str) -> str:
    """按 (deviceSn, sessionId) 稳定派生匿名 contactId。"""
    seed = f"{device_sn}:{session_id}"
    return ANON_PREFIX + str(uuid.uuid3(uuid.NAMESPACE_DNS, seed))

def is_anonymous(contact_id: str) -> bool:
    return contact_id and contact_id.startswith(ANON_PREFIX)
```

**学习要点**:
- LangGraph checkpoint 机制
- 向量检索 (FAISS)
- 多用户隔离

---

### 3.5 Agent 核心 (`agent/`)

#### 3.5.1 Agent 管理器 (`agent/manager.py`)

**行为**:
- 按 `deviceSn` 懒加载 Agent 实例
- 注册表 + 热重载(swap 模式)

**Python 实现**:

```python
class AgentManager:
    """Agent 生命周期管理,按 device_sn 懒加载。"""

    def __init__(self, config_loader, llm_factory, tool_factory,
                 soul_orchestrator, reme_store):
        self._registry: dict[str, CompiledGraph] = {}

    def get_agent(self, device_sn: str) -> CompiledGraph:
        return self._registry.setdefault(device_sn,
            self._build_agent(device_sn))

    def _build_agent(self, device_sn: str) -> CompiledGraph:
        agent_config = self.config_loader.get_agent(device_sn)
        llm = self.llm_factory.create(agent_config.model_id)
        tools = self.tool_factory.build(agent_config)

        graph = build_agent_graph(
            llm, tools, soul_orchestrator, reme_store,
        )
        return graph.compile(checkpointer=MemorySaver())
```

#### 3.5.2 LangGraph 图定义 (`agent/graph.py`)

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    device_sn: str
    contact_id: str
    session_id: str
    messages: list[BaseMessage]
    soul_context: SoulContext
    # ...

def build_agent_graph(llm, tools, soul_orchestrator, reme_store):
    graph = StateGraph(AgentState)

    # 节点
    graph.add_node("prepare_soul", prepare_soul_node)
    graph.add_node("retrieve_reme", retrieve_reme_node)
    graph.add_node("react_agent", create_react_agent(llm, tools))
    graph.add_node("record_reme", record_reme_node)
    graph.add_node("finalize_soul", finalize_soul_node)

    # 边: 线性链路
    graph.set_entry_point("prepare_soul")
    graph.add_edge("prepare_soul", "retrieve_reme")
    graph.add_edge("retrieve_reme", "react_agent")
    graph.add_edge("react_agent", "record_reme")
    graph.add_edge("record_reme", "finalize_soul")
    graph.add_edge("finalize_soul", END)

    return graph
```

**学习要点**:
- `StateGraph` 定义
- `add_node` / `add_edge`
- `create_react_agent` 预构建 ReAct
- `compile(checkpointer=)` 持久化
- `thread_id` 多会话隔离

---

### 3.6 API 层 (`api/`)

#### 3.6.1 FastAPI 服务 (`api/server.py`)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI(title="langchain-ayaka")

@app.post("/agent/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE 流式对话。"""
    return StreamingResponse(
        _sse_generator(request),
        media_type="text/event-stream",
    )

@app.post("/agent/chat/sync")
async def chat_sync(request: ChatRequest):
    """同步对话。"""
    agent = agent_manager.get_agent(request.device_sn)
    config = {"configurable": {"thread_id": request.session_id}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(request.message)], ...},
        config=config,
    )
    return {"content": result["messages"][-1].content}
```

#### 3.6.2 SSE 事件映射 (`api/sse.py`)

**SSE 事件**:
- `token` — 文本增量
- `thinking` — 思考过程
- `tool_call` — 工具调用开始
- `tool_result` — 工具结果
- `done` — 对话结束
- `error` — 异常

```python
async def _sse_generator(request: ChatRequest):
    agent = agent_manager.get_agent(request.device_sn)
    config = {"configurable": {"thread_id": request.session_id}}

    async for event in agent.astream_events(
        {"messages": [HumanMessage(request.message)], ...},
        config=config, version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield _sse("token", {"type": "token", "data": chunk.content})
        elif kind == "on_tool_start":
            yield _sse("tool_call", {"type": "tool_call",
                        "toolName": event["name"]})
        elif kind == "on_tool_end":
            yield _sse("tool_result", {"type": "tool_result",
                        "toolName": event["name"]})

    yield _sse("done", {"type": "done", "sessionId": request.session_id})
```

**学习要点**:
- FastAPI + SSE
- `astream_events` API
- `thread_id` checkpoint

---

### 3.7 意图快捷 (`intent/`)

**行为**: 命中快捷意图(问时间/计算)时,工具结果预填进消息,绕过 ReAct 决策。

```python
class IntentShortcutEnhancer:
    def enhance(self, message: str) -> str:
        """命中快捷意图时增强消息,否则原样返回。"""
        rule = self.classifier.classify(message)
        if rule is None:
            return message
        tool_result = self.tool_registry.invoke(rule.intent_type, message)
        if not tool_result:
            return message
        return rule.enhance_template.format(
            message=message, tool_result=tool_result)
```

---

## 4. 配置规格

### 4.1 Agent 配置 (`config/agents/default.yaml`)

```yaml
device_sn: "default"
name: "ayaka"
owner_id: "1"
model_id: 1  # 引用 config/models/
sys_prompt: |
  你是ayaka,一个温暖、有情绪的陪伴型 AI。
  你有血有肉,会开心、会着急、会委屈、会心疼。
  禁止说"我是机器人""我没有真实情绪"。
max_iters: 10
compaction_keep_messages: 50

# 工具白名单/黑名单
tools_allow: ["current_time", "calculator", "knowledge_search"]
tools_deny: []

# 知识库
knowledge_ids: [1, 2]

# 人格
persona_type: "WARM_COMPANION"
personality_dimensions:
  empathy: 0.55
  dominance: 0.45
  formality: 0.30
  curiosity: 0.50
  relationship_distance: 0.45
  humor: 0.40
  drive_connection: 0.55
  drive_novelty: 0.50
  drive_expression: 0.50
  drive_safety: 0.70
  drive_play: 0.35
  baseline_valence: 0.15
  baseline_arousal: 0.0
  baseline_dominance: 0.0
```

### 4.2 人格策略配置 (`config/personality/strategies.yaml`)

```yaml
WARM_COMPANION:
  behavior_weights:
    warmth:
      empathy: 0.30
      attachment: 0.25
      trust: 0.20
      driveConnection: 0.25
    initiative:
      dominance: 0.35
      driveConnection: 0.25
      boredom: 0.15
      inverseDriveSafety: 0.25
    curiosity:
      curiosity: 0.60
      driveNovelty: 0.40
    restraint:
      driveSafety: 0.45
      formality: 0.10
      frustration: 0.35
      inverseTrust: 0.10
    playfulness:
      humor: 0.25
      drivePlay: 0.45
      valenceNormalized: 0.30
    actionDesire:
      driveExpression: 0.50
      arousalNormalized: 0.25
      playfulness: 0.20
      driveSafety: 0.10
  emotion_deltas:
    ANGRY: {valence: -0.35, arousal: 0.25, dominance: -0.15, trust: -0.08, attachment: 0.0, frustration: 0.18}
    NEGATIVE_FEEDBACK: {valence: -0.25, arousal: 0.10, dominance: -0.12, trust: -0.06, attachment: 0.0, frustration: 0.14}
    GRATEFUL: {valence: 0.24, arousal: 0.05, dominance: 0.10, trust: 0.08, attachment: 0.03, frustration: -0.03}
    SAD: {valence: -0.32, arousal: -0.08, dominance: 0.0, trust: 0.0, attachment: 0.06, frustration: 0.02}
    ANXIOUS: {valence: -0.24, arousal: 0.20, dominance: 0.0, trust: 0.0, attachment: 0.04, frustration: 0.05}
    POSITIVE: {valence: 0.26, arousal: 0.12, dominance: 0.08, trust: 0.03, attachment: 0.02, frustration: -0.02}
    CONFIDING: {valence: -0.08, arousal: -0.02, dominance: 0.0, trust: 0.01, attachment: 0.05, frustration: 0.0}
    NEUTRAL: {}
  state:
    decay_rate_min: 0.05
    decay_rate_max: 0.35
    tool_success_trust_delta: 0.03
    attachment_increment: 0.01
  trigger:
    action_desire_threshold: 0.70
    care_warmth_threshold: 0.75
```

---

## 5. 实施阶段

### 阶段 1: 项目骨架 + LLM + 基础对话

**目标**: 跑通最小闭环 — 用户发消息 → LLM 回复 → SSE 流式输出

**交付物**:
- `pyproject.toml` + 依赖
- `config/models/qwen.yaml`
- `llm/factory.py`
- `agent/manager.py` (简化版: 无灵魂/记忆)
- `agent/graph.py` (简化版: 只有 `react_agent` 节点)
- `api/server.py` + `api/sse.py`
- `scripts/run.py`
- 验证: `curl` 发消息,收到 SSE 流式回复

**学习点**: LangChain ChatModel, LangGraph 基础, SSE

### 阶段 2: 工具系统

**目标**: Agent 能调用工具

**交付物**:
- `tools/builtin.py` (current_time, calculator)
- `tools/mcp.py` (LOCAL/REMOTE)
- `tools/rest.py`
- `tools/knowledge.py`
- `config/tools/` 配置
- 验证: 问"现在几点"→ 调 current_time; 问"3+5"→ 调 calculator

**学习点**: `@tool`, `StructuredTool`, MCP, `args_schema`

### 阶段 3: 人格系统

**目标**: Agent 有人格,回复风格随人格/情绪/关系变化

**交付物**:
- `soul/models.py` (全部数据模型)
- `soul/emotion.py` (EmotionAnalyzer)
- `soul/behavior.py` (BehaviorSignalGenerator)
- `soul/state_machine.py`
- `soul/prompt_composer.py`
- `soul/orchestrator.py` (Preparer)
- `config/personality/` 配置
- `agent/graph.py` 增加 `prepare_soul` 节点
- 验证: 同一消息,不同人格 → 不同回复风格

**学习点**: LangGraph 自定义节点, State 传递, Prompt 模板

### 阶段 4: 长期记忆

**目标**: Agent 记住跨会话的信息

**交付物**:
- `memory/reme_store.py`
- `memory/anonymous.py`
- `agent/graph.py` 增加 `retrieve_reme` + `record_reme` 节点
- 验证: 第一轮说"我叫张三",第二轮问"我叫什么"→ 记得

**学习点**: LangGraph checkpoint, 向量检索, 多用户隔离

### 阶段 5: 灵魂收尾 + 收官

**目标**: 状态机更新 + 多会话隔离 + 意图快捷

**交付物**:
- `soul/orchestrator.py` 补充 Finalizer
- `agent/graph.py` 增加 `finalize_soul` 节点
- `intent/enhancer.py`
- 多会话测试
- 验证: 多轮对话后状态机正确转移; 切换 session 历史隔离

**学习点**: 异步副作用, 状态持久化, 完整闭环

---

## 6. 关键设计决策

### 6.1 为什么用 LangGraph 而不是裸 AgentExecutor?

系统的处理链是**线性预处理 + ReAct + 后处理**结构,LangGraph 的 `StateGraph` 天然表达:
- 预处理节点: `prepare_soul`, `retrieve_reme`
- ReAct: `create_react_agent`
- 后处理: `record_reme`, `finalize_soul`

裸 `AgentExecutor` 无法优雅表达这个链路。

### 6.2 为什么用 YAML 配置?

1. 学习阶段不依赖外部服务,降低环境复杂度
2. YAML 可版本控制,清晰可见
3. 2 核 2G 服务器跑 SQLite 比独立数据库省资源

### 6.3 State 怎么在 LangGraph 节点间传递?

LangGraph 的 `StateGraph` 用 `TypedDict` 定义 state,节点返回的 dict 会 **merge** 进 state。

```python
class AgentState(TypedDict):
    device_sn: str
    messages: list[BaseMessage]
    soul_context: SoulContext

def prepare_soul_node(state: AgentState) -> dict:
    soul_ctx = orchestrator.prepare(state["device_sn"], ...)
    return {"soul_context": soul_ctx}  # merge 进 state
```

### 6.4 多会话隔离怎么实现?

LangGraph 的 `checkpointer` + `thread_id`:
- `thread_id = session_id` → 每个会话独立 checkpoint
- `config = {"configurable": {"thread_id": session_id}}`
- 同一 session 的多轮对话共享历史

### 6.5 灵魂状态(SoulState)存哪?

- 阶段 1-3: 内存 dict(重启丢失,够用)
- 阶段 4+: SQLite
- 可选: LangGraph `SqliteSaver` 直接持久化

### 6.6 2 核 2G 部署可行性

```
Python 进程 + LangChain + LangGraph + FastAPI  ~400MB
FAISS 向量库(小规模)                            ~50MB
SQLite + 本地文件                                ~20MB
──────────────────────────────────────────────
总占用                                          ~470MB
```

2G 内存(可用 ~1.7G)绰绰有余。LLM 走远程 API 不占本地资源。

---

## 7. 验收标准

| 阶段 | 验证方式 | 预期结果 |
|------|---------|---------|
| 1 | `curl POST /agent/chat/stream` | SSE 流式返回 token |
| 2 | 问"现在几点" | 调 current_time 工具,返回时间 |
| 3 | 同一消息,切换人格 | 回复风格变化 |
| 4 | 跨会话问之前信息 | 记得之前说的内容 |
| 5 | 多轮对话后查状态 | PAD/trust 正确转移 |

---

## 8. 参考资料

### LangChain/LangGraph 文档

- LangChain: https://python.langchain.com/docs/
- LangGraph: https://langchain-ai.github.io/langgraph/
- MCP Adapters: https://github.com/langchain-ai/langchain-mcp-adapters

---

## 附录 A: 情绪词表与 deltas 完整对照

| 情绪 | 关键词 | valence | arousal | dominance | trust | attachment | frustration |
|------|--------|---------|---------|-----------|-------|------------|-------------|
| ANGRY | 生气/气死/垃圾/烦/太差 | -0.35 | 0.25 | -0.15 | -0.08 | 0 | 0.18 |
| NEGATIVE_FEEDBACK | 不对/错了/没用/不行/失败 | -0.25 | 0.10 | -0.12 | -0.06 | 0 | 0.14 |
| GRATEFUL | 谢谢/感谢/辛苦/帮了我 | 0.24 | 0.05 | 0.10 | 0.08 | 0.03 | -0.03 |
| SAD | 难过/伤心/委屈/失落/哭/累... | -0.32 | -0.08 | 0 | 0 | 0.06 | 0.02 |
| ANXIOUS | 焦虑/担心/害怕/压力/怎么办... | -0.24 | 0.20 | 0 | 0 | 0.04 | 0.05 |
| POSITIVE | 开心/喜欢/太好了/棒/哈哈/不错 | 0.26 | 0.12 | 0.08 | 0.03 | 0.02 | -0.02 |
| CONFIDING | (长文本含 SAD/ANXIOUS 词) | -0.08 | -0.02 | 0 | 0.01 | 0.05 | 0 |
| NEUTRAL | (无命中) | 0 | 0 | 0 | 0 | 0 | 0 |

## 附录 B: 行为信号权重矩阵

| 信号 | 维度1 | 维度2 | 维度3 | 维度4 |
|------|-------|-------|-------|-------|
| warmth | empathy×0.30 | attachment×0.25 | trust×0.20 | driveConnection×0.25 |
| initiative | dominance×0.35 | driveConnection×0.25 | boredom×0.15 | (1-driveSafety)×0.25 |
| curiosity | curiosity×0.60 | driveNovelty×0.40 | - | - |
| restraint | driveSafety×0.45 | formality×0.10 | frustration×0.35 | (1-trust)×0.10 |
| playfulness | humor×0.25 | drivePlay×0.45 | normalize(valence)×0.30 | - |
| actionDesire | driveExpression×0.50 | normalize(arousal)×0.25 | playfulness×0.20 | driveSafety×0.10 |

## 附录 C: 关系档位推断阈值

| 档位 | trust | attachment | turn_count |
|------|-------|------------|------------|
| STRANGER | < 0.65 | < 0.35 | < 3 |
| FAMILIAR | ≥0.65 或 ≥0.35 或 ≥3 | | |
| TRUSTED | ≥0.78 | ≥0.68 | - |
| INTIMATE | ≥0.88 | ≥0.82 | ≥30 |
| FAMILY_COMPANION | (仅 profile 决定) | | |
