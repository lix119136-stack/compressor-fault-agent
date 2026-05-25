import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import matplotlib.pyplot as plt
from openai import OpenAI
from model_structure import Compressor1DCNN

# =====================================================================
# 1. 网页基础配置与资源加载
# =====================================================================
st.set_page_config(
    page_title="往复压缩机示功图智能诊断系统",
    page_icon="⚙️",
    layout="wide"
)

# ---- 【核心优化】启用 Matplotlib 的矢量通用数学字体，彻底免疫系统缺字引发的方块乱码 ----
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'stix'


@st.cache_resource
def load_diagnostic_resources():
    """使用 Streamlit 缓存机制，确保模型和客户端只加载一次"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("data_scaler.pkl")

    model = Compressor1DCNN(num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load("compressor_cnn_model.pth", map_location=device))
    model.to(device)
    model.eval()

    client = OpenAI(
        api_key=st.secrets["deepseek_key"],
        base_url="https://api.deepseek.com/v1"
    )
    return model, label_encoder, scaler, device, client


try:
    model, label_encoder, scaler, device, client = load_diagnostic_resources()
except Exception as e:
    st.error(f"❌ 智能体核心模型组件加载失败，请检查配置文件。错误原因: {e}")

# 大模型专家级 System Prompt（要求大模型用中文输出极其详尽的报告）
SYSTEM_PROMPT = """你是一位精通机械工程、热力学与流体力学、专攻往复压缩机故障诊断的【金牌 AI 智能体】。
用户会上传单周期示功图数据，底层神经网络（1D-CNN）和热力学几何计算工具会返回结构化指标。
你的任务是：根据这些量化指标，撰写一份结构严谨、逻辑闭环的工业级中文技术报告。
报告规范：必须明确包含【诊断结论】、【数据定量指标】、【热力学机理分析】和【维修消缺建议】。
请严格使用 Markdown 语法输出，多使用加粗进行强调，整体文风要求专业、硬核。"""

# =====================================================================
# 2. 网页前端布局设计（保持全中文的人性化交互）
# =====================================================================
st.title("⚙️ 往复压缩机故障诊断智能体系统 (Web 3.0 终极免依赖版)")
st.caption("山东省自然科学基金资助项目 (ZR2024ME123) · 深度学习与大模型协同诊断平台")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 数据输入中心")
    uploaded_file = st.file_uploader("请直接拖拽或选择压缩机示功图 CSV 文件", type=["csv"])

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
        <strong>💡 使用说明:</strong><br>
        1. 准备单周期往复压缩机示功图数据（CSV格式）。<br>
        2. 数据前两列需满足：<strong>第一列为容积 (V)，第二列为压力 (P)</strong>。<br>
        3. 系统将自动调用 <strong>1D-CNN 神经网络</strong> 进行多类故障模式识别，并联动 <strong>DeepSeek 大模型</strong> 实时秒级流式生成专家技术报告。
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("📊 实时诊断与机理报告")

    tool_result = None

    if uploaded_file is not None:
        with st.spinner("⚡ 智能体正在解析数据并启动 1D-CNN 推理引擎..."):
            try:
                df = pd.read_csv(uploaded_file, header=None, usecols=[0, 1])
                if df.isna().sum().sum() > 0:
                    df = df.ffill().bfill()

                volume = df.iloc[:, 0].values
                pressure = df.iloc[:, 1].values

                # 定量热力学指标提取
                max_p = float(np.max(pressure))
                min_p = float(np.min(pressure))
                indicator_work = float(
                    0.5 * np.abs(np.dot(volume, np.roll(pressure, 1)) - np.dot(pressure, np.roll(volume, 1))))

                # 神经网络推理
                raw_feature = df.values.flatten().reshape(1, -1)
                scaled_feature = scaler.transform(raw_feature)
                tensor_feature = torch.tensor(scaled_feature, dtype=torch.float32).unsqueeze(1).to(device)

                with torch.no_grad():
                    outputs = model(tensor_feature)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                pred_class_idx = int(np.argmax(probabilities))
                pred_label = label_encoder.inverse_transform([pred_class_idx])[0]
                confidence = float(probabilities[pred_class_idx])

                # 打包作为大模型的感知输入
                tool_result = {
                    "diagnosis": pred_label,
                    "confidence": f"{confidence * 100:.2f}%",
                    "metrics": {
                        "maximum_pressure": round(max_p, 2),
                        "minimum_pressure": round(min_p, 2),
                        "indicator_work_area": round(indicator_work, 2)
                    }
                }

                # ---- 【国际化 SCI 级图表渲染】单独使用学术标准英文，100% 杜绝方块码 ----
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(volume, pressure, 'r-', linewidth=2, label="Measured p-V Curve")

                # 采用标准科技论文格式配置标签
                ax.set_title(f"Real-time p-V Diagram ({pred_label})", fontsize=11, fontweight='bold')
                ax.set_xlabel("Volume V", fontsize=10, fontstyle='italic')
                ax.set_ylabel("Pressure P", fontsize=10, fontstyle='italic')
                ax.legend(loc="upper right")
                ax.grid(True, linestyle="--", alpha=0.5)

                plt.tight_layout()
                st.pyplot(fig)  # 稳稳输出图表

                # 网页交互提示仍保持亲切的全中文
                st.success(f"🎉 神经网络识别完成：检测到【{pred_label}】，置信度达 {confidence * 100:.2f}%")

            except Exception as e:
                st.error(f"❌ 感知计算失败，请确认导入的 CSV 数据格式。错误原因: {e}")

        # ---- 阶段 2：大模型流式技术报告生成 ----
        if tool_result is not None:
            st.markdown("---")
            st.markdown("### 🧠 DeepSeek 专家级诊断技术报告")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"感知计算工具链返回的最新状态特征数据如下:\n{json.dumps(tool_result, ensure_ascii=False, indent=4)}"}
            ]

            try:
                report_container = st.empty()
                full_response = ""

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.2,
                    stream=True
                )

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        report_container.markdown(full_response + "▌")

                report_container.markdown(full_response)

            except Exception as llm_err:
                st.error(f"❌ 专家技术报告生成中断，请尝试刷新重试。错误原因: {llm_err}")
    else:
        st.info("📢 暂无待处理数据。请在左侧上传一个示功图 CSV 文件，智能体将实时为您诊断。")