import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import json
import matplotlib.pyplot as plt
from openai import OpenAI
from model_structure import Compressor1DCNN
# 确保 app.py 顶部只有这一行引入网络结构
from model_structure import Compressor1DCNN

# =====================================================================
# 1. 网页基础配置与资源加载
# =====================================================================
st.set_page_config(page_title="往复压缩机智能诊断系统", layout="wide")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


@st.cache_resource
def load_diagnostic_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("data_scaler.pkl")

    model = Compressor1DCNN(num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load("compressor_cnn_model.pth", map_location=device))
    model.to(device)
    model.eval()

    client = OpenAI(
        api_key="sk-5dfe1af7504e482fb52265a4fb872619",
        base_url="https://api.deepseek.com/v1"
    )
    return model, label_encoder, scaler, device, client


try:
    model, label_encoder, scaler, device, client = load_diagnostic_resources()
except Exception as e:
    st.error(f"加载模型组件失败: {e}")

# 大模型 System Prompt
SYSTEM_PROMPT = """你是一位精通机械工程和热力学机理的【往复压缩机故障诊断专家智能体】。
请根据底层工具链提供的诊断结果（包含类别、置信度、压力极值和指示功面积），撰写一份格式严谨的工业级技术报告。
报告规范：必须包含【诊断结论】、【数据定量指标】、【热力学机理分析】和【维修消缺建议】。请使用 Markdown 语法输出，重点词汇加粗。"""

# =====================================================================
# 2. 网页布局设计
# =====================================================================
st.title("⚙️ 往复压缩机故障诊断智能体系统")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 数据输入中心")
    uploaded_file = st.file_uploader("请直接拖拽或选择压缩机示功图 CSV 文件", type=["csv"])

    st.markdown("""
    **💡 使用说明:**
    1. 准备单周期往复压缩机示功图数据（CSV格式）。
    2. 第一列需为**容积 (V)**，第二列需为**压力 (P)**。
    3. 系统将自动调用 **1D-CNN 神经网络** 进行模式识别，并联动 **DeepSeek** 生成专家报告。
    """)

with col2:
    st.subheader("📊 实时诊断与机理报告")

    # 定义全局变量，确保数据生命周期安全
    tool_result = None
    volume, pressure = None, None
    pred_label, confidence = "", 0.0

    if uploaded_file is not None:
        # ---- 核心优化阶段 1：静态组件一次性渲染完毕（不与大模型并行） ----
        with st.spinner("⚡ 神经网络正在提取波形模式..."):
            try:
                df = pd.read_csv(uploaded_file, header=None, usecols=[0, 1])
                if df.isna().sum().sum() > 0:
                    df = df.ffill().bfill()

                volume = df.iloc[:, 0].values
                pressure = df.iloc[:, 1].values

                # 特征计算
                max_p = float(np.max(pressure))
                min_p = float(np.min(pressure))
                indicator_work = float(
                    0.5 * np.abs(np.dot(volume, np.roll(pressure, 1)) - np.dot(pressure, np.roll(volume, 1))))

                # 神经网络模式识别
                raw_feature = df.values.flatten().reshape(1, -1)
                scaled_feature = scaler.transform(raw_feature)
                tensor_feature = torch.tensor(scaled_feature, dtype=torch.float32).unsqueeze(1).to(device)

                with torch.no_grad():
                    outputs = model(tensor_feature)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                pred_class_idx = int(np.argmax(probabilities))
                pred_label = label_encoder.inverse_transform([pred_class_idx])[0]
                confidence = float(probabilities[pred_class_idx])

                tool_result = {
                    "diagnosis": pred_label,
                    "confidence": f"{confidence * 100:.2f}%",
                    "metrics": {
                        "maximum_pressure": round(max_p, 2),
                        "minimum_pressure": round(min_p, 2),
                        "indicator_work_area": round(indicator_work, 2)
                    }
                }

                # 静态绘制图表
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(volume, pressure, 'r-', linewidth=2, label=f"解析波形 ({pred_label})")
                ax.set_title("动态解析 p-V 示功图", fontsize=11, fontweight='bold')
                ax.set_xlabel("容积 (V)")
                ax.set_ylabel("压力 (P)")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                st.pyplot(fig)

                st.success(f"神经网络识别完成：检测到【{pred_label}】，置信度 {confidence * 100:.2f}%")

            except Exception as e:
                st.error(f"底层分析失败，错误原因: {e}")

        # ---- 核心优化阶段 2：大模型流式输出（放在独立的安全区间内） ----
        if tool_result is not None:
            st.markdown("---")
            st.markdown("### 🧠 DeepSeek 专家级诊断技术报告")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"分析请求:\n{json.dumps(tool_result, ensure_ascii=False, indent=4)}"}
            ]

            try:
                # 建立全新的空全局占位符
                report_container = st.empty()
                full_response = ""

                # 发起大模型流式请求
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.2,
                    stream=True
                )

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        # 采用专门的空容器实时更新文本，避开 DOM 组件竞争
                        report_container.markdown(full_response + "▌")

                # 输出完毕后移除打字机光标
                report_container.markdown(full_response)

            except Exception as llm_err:
                st.error(f"流式报告生成中断，请重试。错误: {llm_err}")
    else:
        st.info("📢 暂无待处理数据。请在左侧上传一个示功图 CSV 文件，智能体将实时为您诊断。")