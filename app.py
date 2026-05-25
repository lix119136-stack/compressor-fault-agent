import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os  # 用于路径动态解析
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 用于精细化控制中文字体加载
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

# ---- 【完美兼容大写解法】动态锁定本地字体，支持 .ttf 和 .TTF 后缀 ----
my_font = None
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 尝试匹配大写后缀（完美对齐你 GitHub 上的 simhei.TTF）
    font_path_upper = os.path.join(current_dir, "simhei.TTF")
    font_path_lower = os.path.join(current_dir, "simhei.ttf")

    if os.path.exists(font_path_upper):
        my_font = fm.FontProperties(fname=font_path_upper)
    elif os.path.exists(font_path_lower):
        my_font = fm.FontProperties(fname=font_path_lower)
    else:
        # 如果实在找不到，尝试默认盲读
        my_font = fm.FontProperties(fname="simhei.TTF")
except Exception as f_err:
    my_font = None


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

# 大模型专家级 System Prompt
SYSTEM_PROMPT = """你是一位精通机械工程、热力学与流体力学的【往复压缩机故障诊断金牌智能体】。
请根据底层工具链提供的诊断结果（包含故障类别、置信度、压力极值和指示功面积），撰写一份格式严谨的工业级技术报告。
报告规范：必须包含【诊断结论】、【数据定量指标】、【热力学机理分析】和【维修消缺建议】。
请严格使用 Markdown 语法输出，多使用加粗进行强调，文字风格要专业、严谨。"""

# =====================================================================
# 2. 网页前端布局设计
# =====================================================================
st.title("⚙️ 往复压缩机故障诊断智能体系统 (Web 3.0 全中文完全体)")
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

                # 特征指标计算
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

                # ---- 【全中文图表配置区域】核心代码修改 ----
                fig, ax = plt.subplots(figsize=(6, 3.5))

                if my_font:
                    # 成功捕获本地 simhei.TTF，渲染全中文科研图表
                    ax.plot(volume, pressure, 'r-', linewidth=2, label="实测 p-V 示功图曲线")
                    ax.set_title(f"实时动态示功图 - 诊断结果：{pred_label}", fontproperties=my_font, fontsize=12, fontweight='bold')
                    ax.set_xlabel("容积 Volume (V)", fontproperties=my_font, fontsize=10)
                    ax.set_ylabel("压力 Pressure (P)", fontproperties=my_font, fontsize=10)
                    ax.legend(prop=my_font, loc="upper right")
                else:
                    # 极端防崩溃后备方案
                    ax.plot(volume, pressure, 'r-', linewidth=2, label="Measured p-V Curve")
                    ax.set_title(f"Real-time p-V Diagram ({pred_label})", fontsize=11, fontweight='bold')
                    ax.set_xlabel("Volume V")
                    ax.set_ylabel("Pressure P")
                    ax.legend(loc="upper right")

                ax.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                st.pyplot(fig)  # 将带有中文图例的图表推向前端

                st.success(f"🎉 神经网络识别完成：检测到【{pred_label}】，置信度达 {confidence * 100:.2f}%")

            except Exception as e:
                st.error(f"❌ 感知计算失败，请确认导入的 CSV 数据格式。错误原因: {e}")

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