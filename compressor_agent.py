import os
import json
import re
from agent_diagnostic_tool import CompressorAgentTool
from openai import OpenAI

client = OpenAI(
    api_key="sk-5dfe1af7504e482fb52265a4fb872619",
    base_url="https://api.deepseek.com/v1"
)

# 初始化升级后的核心诊断工具
diagnostic_tool = CompressorAgentTool()

# =====================================================================
# 升级版：智能体专家提示词（强化数据定量分析要求）
# =====================================================================
SYSTEM_PROMPT = """你是一位精通机械工程、热力学与流体力学的【往复压缩机故障诊断金牌智能体】。
你拥有高精度的底层神经网络感知工具以及定量热力学几何计算工具。
当用户输入数据路径时，工具会返回诊断类别、置信度、压力极值，以及核心的【示功图封闭面积（即指示功 Area）】。

你的任务是：
1. 结合工具返回的真实量化指标（如最大压力、最小压力、指示功面积），在报告中进行【定量与定性结合】的专业技术分析。
   - 必须在报告中引用返回的 `indicator_work_area` 数值，并从热力学功的消耗角度（如阻力功增加、泄漏导致膨胀功变异等）剖析该数值的物理意义。
2. 解释示功图在当前数值表现下的几何畸变机理。
3. 明确指明可视化分析图表已经成功渲染并保存的本地绝对路径。
4. 提供工业级、具备落地实操价值的故障消除与维护建议。

请统一使用严谨的工业技术报告格式输出，包含：【诊断结论】、【数据定量指标】、【热力学机理分析】、【可视化图表引导】和【维修消缺建议】。"""


def get_llm_response(user_query, tool_result):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"用户提问: {user_query}\n\n底层感知与定量计算工具链返回的真实数据如下:\n{json.dumps(tool_result, ensure_ascii=False, indent=4)}"}
    ]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"大模型生成诊断报告失败，错误原因: {str(e)}"


# =====================================================================
# 智能体交互主循环
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🤖 往复压缩机故障诊断智能体 (Agent 2.0 完全体) 已成功启动！")
    print("输入样例：分析一下 E:\\课题数据\\示功图\\排气阀泄漏\\300.458-97.659-3.07--2.4--1.4.csv")
    print("=" * 50 + "\n")

    while True:
        user_input = input("用户 >> ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("智能体已关闭，祝您科研顺利！")
            break

        if not user_input:
            continue

        path_match = re.search(r'[a-zA-Z]:\\[^\s]+', user_input)

        if path_match:
            csv_path = path_match.group(0)
            print(f"⚙️ 智能体正在启动 1D-CNN 推理引擎并运行热力学积分算法...")

            # 1. 调用升级后的工具链（同时完成分类、积分计算、画图保存三项任务）
            tool_res = diagnostic_tool.diagnose_csv(csv_path)

            if tool_res["status"] == "success":
                print("🧠 正在调用 DeepSeek 解析定量指标并生成高级热力学技术报告...")
                expert_report = get_llm_response(user_input, tool_res)
                print("\n" + "-" * 60)
                print(expert_report)
                print("-" * 60 + "\n")
            else:
                print(f"❌ 智能体运行中止: {tool_res['message']}")
        else:
            print("🤖 智能体提示：请输入包含有效 CSV 文件绝对路径的指令。")