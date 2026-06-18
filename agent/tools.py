"""
WorkGuard-VL 工具注册表
=====================

本模块实现了 Agent 的工具调用（Tool Calling / Function Calling）机制。

什么是 Tool Calling？
    LLM 本身只能生成文本，不能执行真实操作（查数据库、发消息等）。
    Tool Calling 让 LLM 可以"请求"调用外部工具，由我们的代码执行后把结果返回给 LLM。

    工作流程：
    ┌────────┐   "我要查工位A1的员工信息"              ┌────────┐
    │  LLM   │ ──────────────────────────────────→ │  Agent │
    └────────┘   (输出 ACTION: query_employee(A1))  └────┬───┘
         ↑                                               │
         │  "员工张三，研发部，算法工程师"                   │ 执行工具
         │ ←──────────────────────────────────────────────┘
         │
    ┌────────┐
    │  LLM   │ 基于工具返回结果，生成最终回答
    └────────┘

为什么手写而不依赖 LangChain？
    1. 简历上可以写"手写工具注册表和调度器"，比"集成 LangChain"更有含金量
    2. 代码量小（~300行），逻辑清晰，面试时能讲清楚每个细节
    3. 不引入重型依赖，部署更简单

设计模式：
    - 装饰器模式（@tool）：声明工具的名称、描述、参数
    - 注册表模式（ToolRegistry）：统一管理所有工具
    - 策略模式：每个工具是一个独立的函数，可插拔
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agent.rag import RAGPipeline


# ============================================================================
# 1. 工具数据结构
# ============================================================================

@dataclass
class ToolParam:
    """
    工具参数定义。

    属性说明：
        name: 参数名（如 "workstation_id"）
        type: 参数类型（如 "str"、"int"、"float"）
        description: 参数描述（给 LLM 看的，帮助它理解如何填这个参数）
        required: 是否必填
        default: 默认值（可选）
    """
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolDef:
    """
    工具定义。

    属性说明：
        name: 工具名称（如 "query_employee"），LLM 会用这个名字来调用
        description: 工具描述（给 LLM 看的，帮助它决定何时使用这个工具）
            - 描述越清晰，LLM 越能正确选择工具
            - 例如："查询指定工位的员工信息，返回姓名、部门、岗位"
        func: 实际执行的函数
        params: 参数定义列表
    """
    name: str
    description: str
    func: Callable[..., Any]
    params: list[ToolParam] = field(default_factory=list)

    def to_schema(self) -> dict[str, Any]:
        """
        生成 OpenAI Function Calling 格式的工具 schema。

        这个 schema 会被塞入 LLM 的 system prompt，告诉 LLM 有哪些工具可用、
        每个工具接受什么参数。

        返回格式（OpenAI function calling 标准格式）：
        {
            "type": "function",
            "function": {
                "name": "query_employee",
                "description": "查询指定工位的员工信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "workstation_id": {
                            "type": "str",
                            "description": "工位ID，如 A1、A2"
                        }
                    },
                    "required": ["workstation_id"]
                }
            }
        }
        """
        properties = {}
        required = []
        for param in self.params:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ============================================================================
# 2. 工具注册表
# ============================================================================

class ToolRegistry:
    """
    工具注册表：管理所有可用工具。

    核心职责：
        1. 注册工具（通过 @tool 装饰器或手动注册）
        2. 生成工具 schema（给 LLM 看）
        3. 执行工具调用（解析 LLM 的请求，调用对应函数）
        4. 格式化工具结果（返回给 LLM）

    使用方式：
        >>> registry = ToolRegistry()
        >>> @registry.tool(description="查询员工信息")
        ... def query_employee(workstation_id: str) -> dict:
        ...     ...
        >>> schema = registry.get_schemas()  # 给 LLM
        >>> result = registry.execute("query_employee", {"workstation_id": "A1"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def tool(
        self,
        description: str,
        params: list[ToolParam] | None = None,
    ) -> Callable:
        """
        装饰器：注册一个工具函数。

        参数说明：
            description: 工具描述（必须清晰，LLM 靠这个决定是否使用）
            params: 参数定义列表（可选，如果不提供会自动从函数签名推断）

        使用示例：
            >>> @registry.tool(
            ...     description="查询指定工位的员工信息，返回姓名、部门、岗位等",
            ...     params=[
            ...         ToolParam("workstation_id", "str", "工位ID，如 A1", required=True),
            ...     ]
            ... )
            ... def query_employee(workstation_id: str) -> dict:
            ...     return {"name": "张三", "department": "研发部"}

        为什么用装饰器？
            - 声明式：函数定义和工具描述放在一起，代码清晰
            - 自动注册：定义完就自动注册到 registry，不需要手动调用
            - 可扩展：以后可以加权限检查、日志记录等，只改装饰器即可
        """
        def decorator(func: Callable) -> Callable:
            # 自动从函数签名推断参数（如果没手动指定）
            tool_params = params or _infer_params(func)

            tool_def = ToolDef(
                name=func.__name__,
                description=description,
                func=func,
                params=tool_params,
            )
            self._tools[func.__name__] = tool_def
            return func
        return decorator

    def register(self, tool_def: ToolDef) -> None:
        """手动注册一个工具（不通过装饰器）。"""
        self._tools[tool_def.name] = tool_def

    def get_schemas(self) -> list[dict[str, Any]]:
        """
        获取所有工具的 schema 列表。

        这个列表会被格式化后塞入 LLM 的 prompt，
        告诉 LLM "你有这些工具可以用"。
        """
        return [tool.to_schema() for tool in self._tools.values()]

    def get_tools_description(self) -> str:
        """
        生成工具列表的文本描述（用于 prompt）。

        返回格式：
            可用工具：
            1. query_employee: 查询指定工位的员工信息
               参数: workstation_id (str, 必填) - 工位ID，如 A1、A2
            2. query_schedule: 查询员工今日排班
               参数: employee_id (str, 必填) - 员工ID，如 E001
        """
        lines = ["可用工具："]
        for i, tool in enumerate(self._tools.values(), 1):
            lines.append(f"{i}. {tool.name}: {tool.description}")
            for param in tool.params:
                required_str = "必填" if param.required else f"可选, 默认={param.default}"
                lines.append(f"   参数: {param.name} ({param.type}, {required_str}) - {param.description}")
        return "\n".join(lines)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        执行一个工具调用。

        参数说明：
            tool_name: 工具名称（如 "query_employee"）
            arguments: 参数字典（如 {"workstation_id": "A1"}）

        返回值：
            dict: 包含执行结果
                - success (bool): 是否成功
                - result (Any): 工具返回值
                - error (str): 错误信息（如果失败）
                - elapsed_s (float): 执行耗时（秒）
        """
        if tool_name not in self._tools:
            return {
                "success": False,
                "error": f"工具 '{tool_name}' 不存在。可用工具: {list(self._tools.keys())}",
                "result": None,
            }

        tool = self._tools[tool_name]

        # 填充默认值
        for param in tool.params:
            if param.name not in arguments and param.default is not None:
                arguments[param.name] = param.default

        # 执行
        start = time.perf_counter()
        try:
            result = tool.func(**arguments)
            elapsed = time.perf_counter() - start
            return {
                "success": True,
                "result": result,
                "elapsed_s": round(elapsed, 4),
            }
        except TypeError as e:
            return {
                "success": False,
                "error": f"参数错误: {e}",
                "result": None,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"执行失败: {type(e).__name__}: {e}",
                "result": None,
            }

    def has_tool(self, name: str) -> bool:
        """检查工具是否存在。"""
        return name in self._tools

    def list_tools(self) -> list[str]:
        """返回所有工具名称列表。"""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)


def _infer_params(func: Callable) -> list[ToolParam]:
    """
    从函数签名自动推断参数定义。

    这是一个便利函数：如果你不想手动写 params 列表，
    可以让代码自动从函数的 type hints 和默认值推断。

    支持的类型映射：
        str → "string"
        int → "integer"
        float → "number"
        bool → "boolean"
        list → "array"
        dict → "object"
    """
    import inspect

    sig = inspect.signature(func)
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    params: list[ToolParam] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        ptype = type_map.get(param.annotation, "string")
        required = param.default is inspect.Parameter.empty
        default = None if required else param.default
        params.append(ToolParam(
            name=name,
            type=ptype,
            description=f"参数 {name}",
            required=required,
            default=default,
        ))
    return params


# ============================================================================
# 3. 工具实现
# ============================================================================
# 下面是具体的工具函数。每个工具：
#   - 接收明确的参数
#   - 返回结构化的 dict/list
#   - 不依赖外部真实 API（用本地 JSON 模拟）
#
# 在真实项目中，这些函数会连接数据库、调用 REST API 等。
# 这里用 JSON 文件模拟，但接口设计与真实场景一致。


def create_default_registry(
    employees_path: str | Path = "data/employees.json",
    schedule_path: str | Path = "data/schedule.json",
    rag: RAGPipeline | None = None,
) -> ToolRegistry:
    """
    创建默认的工具注册表，包含所有预设工具。

    参数说明：
        employees_path: 员工信息 JSON 文件路径
        schedule_path: 排班信息 JSON 文件路径
        rag: RAG Pipeline 实例（用于制度检索工具）

    返回值：
        ToolRegistry: 注册了所有工具的注册表实例
    """
    registry = ToolRegistry()

    # ---- 加载本地数据 ----
    employees_data = json.loads(Path(employees_path).read_text(encoding="utf-8"))
    employees = employees_data.get("employees", employees_data)

    # 建立索引：workstation_id → employee, employee_id → employee
    ws_to_emp: dict[str, list[dict]] = {}
    id_to_emp: dict[str, dict] = {}
    for emp in employees:
        wid = emp.get("workstation_id")
        if wid:
            ws_to_emp.setdefault(wid, []).append(emp)
        eid = emp.get("employee_id")
        if eid:
            id_to_emp[eid] = emp

    schedule_data = json.loads(Path(schedule_path).read_text(encoding="utf-8"))
    schedules = schedule_data.get("schedules", [])
    activity_log = schedule_data.get("activity_log", [])
    schedule_date = schedule_data.get("date", "unknown")

    id_to_schedule: dict[str, dict] = {}
    for s in schedules:
        eid = s.get("employee_id")
        if eid:
            id_to_schedule[eid] = s

    # ---- 告警日志（内存中） ----
    alert_log: list[dict] = []

    # ========================================================================
    # Tool 1: query_employee — 查询工位的员工信息
    # ========================================================================
    @registry.tool(
        description="查询指定工位的员工信息。返回该工位当前/归属的员工姓名、部门、岗位等。",
        params=[
            ToolParam("workstation_id", "string", "工位ID，如 A1、A2", required=True),
        ],
    )
    def query_employee(workstation_id: str) -> dict:
        emps = ws_to_emp.get(workstation_id, [])
        if not emps:
            return {"found": False, "message": f"工位 {workstation_id} 未找到对应员工"}
        return {
            "found": True,
            "workstation_id": workstation_id,
            "employees": [
                {
                    "employee_id": e.get("employee_id"),
                    "name": e.get("name"),
                    "department": e.get("department"),
                    "position": e.get("position"),
                    "manager": e.get("manager"),
                }
                for e in emps
            ],
        }

    # ========================================================================
    # Tool 2: query_schedule — 查询员工排班和考勤
    # ========================================================================
    @registry.tool(
        description="查询指定员工的排班和考勤状态。返回班次、上下班时间、今日状态等。",
        params=[
            ToolParam("employee_id", "string", "员工ID，如 E001、E002", required=True),
        ],
    )
    def query_schedule(employee_id: str) -> dict:
        sched = id_to_schedule.get(employee_id)
        if not sched:
            return {"found": False, "message": f"员工 {employee_id} 的排班信息不存在"}
        return {
            "found": True,
            "date": schedule_date,
            "schedule": sched,
        }

    # ========================================================================
    # Tool 3: send_alert — 发送异常行为告警
    # ========================================================================
    @registry.tool(
        description="发送异常行为告警。当检测到违纪行为（睡觉、长时间玩手机等）时调用。",
        params=[
            ToolParam("workstation_id", "string", "工位ID", required=True),
            ToolParam("activity", "string", "检测到的异常行为，如：睡觉、玩手机", required=True),
            ToolParam("severity", "string", "告警级别：low/medium/high", required=True),
            ToolParam("details", "string", "补充说明", required=False, default=""),
        ],
    )
    def send_alert(
        workstation_id: str,
        activity: str,
        severity: str,
        details: str = "",
    ) -> dict:
        alert = {
            "timestamp": datetime.now().isoformat(),
            "workstation_id": workstation_id,
            "activity": activity,
            "severity": severity,
            "details": details,
            "status": "sent",
        }
        alert_log.append(alert)
        print(f"\n[ALERT] {severity.upper()}: 工位 {workstation_id} - {activity}")
        if details:
            print(f"[ALERT] 详情: {details}")
        return {"sent": True, "alert": alert}

    # ========================================================================
    # Tool 4: query_activity_history — 查询工位历史活动记录
    # ========================================================================
    @registry.tool(
        description="查询指定工位的历史活动记录。返回最近一段时间的行为日志。",
        params=[
            ToolParam("workstation_id", "string", "工位ID", required=True),
            ToolParam("hours", "integer", "查询最近多少小时的记录，默认24小时", required=False, default=24),
        ],
    )
    def query_activity_history(workstation_id: str, hours: int = 24) -> dict:
        # 查找该工位对应的员工
        emps = ws_to_emp.get(workstation_id, [])
        if not emps:
            return {"found": False, "message": f"工位 {workstation_id} 未找到"}

        emp_ids = {e.get("employee_id") for e in emps}
        records = [
            r for r in activity_log
            if r.get("employee_id") in emp_ids
        ]

        # 统计活动分布
        activity_counts: dict[str, int] = {}
        for r in records:
            act = r.get("activity", "未知")
            activity_counts[act] = activity_counts.get(act, 0) + 1

        return {
            "found": True,
            "workstation_id": workstation_id,
            "date": schedule_date,
            "total_records": len(records),
            "activity_counts": activity_counts,
            "recent_records": records[-10:],  # 最近 10 条
        }

    # ========================================================================
    # Tool 5: search_policies — 检索公司制度文档（调用 RAG）
    # ========================================================================
    @registry.tool(
        description="检索公司制度文档。当需要判断行为是否合规、了解公司规定时调用。",
        params=[
            ToolParam("query", "string", "检索关键词或问题，如：工位上睡觉怎么处理", required=True),
            ToolParam("top_k", "integer", "返回结果数量，默认3", required=False, default=3),
        ],
    )
    def search_policies(query: str, top_k: int = 3) -> dict:
        if rag is None:
            return {
                "found": False,
                "message": "RAG 模块未初始化，无法检索制度文档",
            }

        context, results = rag.query(query, top_k=top_k)

        return {
            "found": len(results) > 0,
            "query": query,
            "results_count": len(results),
            "context": context,
            "details": [
                {
                    "source": r["chunk"].source,
                    "section": r["chunk"].section,
                    "score": round(r["score"], 4),
                    "text_preview": r["chunk"].text[:200],
                }
                for r in results
            ],
        }

    # ========================================================================
    # Tool 6: get_current_time — 获取当前时间
    # ========================================================================
    @registry.tool(
        description="获取当前时间。用于判断是否在工作时间内、计算持续时长等。",
    )
    def get_current_time() -> dict:
        now = datetime.now()
        return {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
            "is_workday": now.weekday() < 5,
        }

    return registry
