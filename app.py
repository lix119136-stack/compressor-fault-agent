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
# 配置网页标题、图标以及宽屏现代布局
st.set_page_config(
    page_title="往复压缩机示功图智能诊断系统",
    page_icon="⚙️",
    layout="wide"
)


@st.cache_resource
def load_diagnostic_resources():
    """使用 Streamlit 缓存机制，确保神经网络、归一化工具和 API 客户端在云端只加载一次"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = joblib.load("label_encoder.pkl")
    scaler = joblib.load("data_scaler.pkl")

    # 动态匹配 11 分类的自适应池化网络
    model = Compressor1DCNN(num_classes=len(label_encoder.classes_))
    model.load_state_dict(torch.load("compressor_cnn_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # 从云端高级设置 Secrets 中安全读取你的安全密钥
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
st.title("⚙️ 往复压缩机故障诊断智能体系统 (Web 3.0 终极版)")
st.caption("山东省自然科学基金资助项目 (ZR2024ME123) · 深度学习与大模型协同诊断平台")
st.markdown("---")

# 创建左右对称的协同工作两栏布局
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📁 数据输入中心")
    # 支持拖拽和点击的文件上传组件
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

    # 建立全局变量，确保生存期内的计算数据安全
    tool_result = None

    if uploaded_file is not None:
        # ---- 阶段 1：静态组件与图表一次性渲染完毕（防止与文本流产生 DOM 竞争） ----
        with st.spinner("⚡ 智能体正在解析数据并启动 1D-CNN 推理引擎..."):
            try:
                # 精准提取前两列，前向与后向填充坏点
                df = pd.read_csv(uploaded_file, header=None, usecols=[0, 1])
                if df.isna().sum().sum() > 0:
                    df = df.ffill().bfill()

                volume = df.iloc[:, 0].values
                pressure = df.iloc[:, 1].values

                # 核心定量特征指标提取
                max_p = float(np.max(pressure))
                min_p = float(np.min(pressure))
                # 使用鞋带公式进行封闭示功图面积积分计算（指示功）
                indicator_work = float(
                    0.5 * np.abs(np.dot(volume, np.roll(pressure, 1)) - np.dot(pressure, np.roll(volume, 1))))

                # 神经网络模式识别（转化为 722 维特征送入模型）
                raw_feature = df.values.flatten().reshape(1, -1)
                scaled_feature = scaler.transform(raw_feature)
                tensor_feature = torch.tensor(scaled_feature, dtype=torch.float32).unsqueeze(1).to(device)

                with torch.no_grad():
                    outputs = model(tensor_feature)
                    probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

                pred_class_idx = int(np.argmax(probabilities))
                pred_label = label_encoder.inverse_transform([pred_class_idx])[0]
                confidence = float(probabilities[pred_class_idx])

                # 打包结构化数据字典，作为大模型的“热力学感知输入”
                tool_result = {
                    "diagnosis": pred_label,
                    "confidence": f"{confidence * 100:.2f}%",
                    "metrics": {
                        "maximum_pressure": round(max_p, 2),
                        "minimum_pressure": round(min_p, 2),
                        "indicator_work_area": round(indicator_work, 2)
                    }
                }

                # ---- 完美修复：全面采用国际化学术英文渲染图表，彻底杜绝云端中文方块乱码 ----
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(volume, pressure, 'r-', linewidth=2, label="Measured p-V Curve")

                ax.set_title(f"Real-time Indicator Diagram ({pred_label})", fontsize=11, fontweight='bold')
                ax.set_xlabel("Volume (V)", fontsize=10)
                ax.set_ylabel("Pressure (P)", fontsize=10)

                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(loc="upper right")

                plt.tight_layout()
                st.pyplot(fig)  # 渲染图表

                # 前端弹出汉化成功的成功状态框
                st.success(f"🎉 神经网络识别完成：检测到【{pred_label}】，置信度达 {confidence * 100:.2f}%")

            except Exception as e:
                st.error(f"❌ 感知计算失败，请确认导入的 CSV 数据格式。错误原因: {e}")

        # ---- 阶段 2：大模型高级机理流式技术报告（在完全隔离的安全容器内渲染） ----
        if tool_result is not None:
            st.markdown("---")
            st.markdown("### 🧠 DeepSeek 专家级诊断技术报告")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"感知计算工具链返回的最新状态特征数据如下:\n{json.dumps(tool_result, ensure_ascii=False, indent=4)}"}
            ]

            try:
                # 建立全局空容器，提供极佳的打字机流式交互特效
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
                        # 动态实时更新文本容器，并附带优雅的输入光标
                        report_container.markdown(full_response + "▌")

                # 输出完毕后移除打字机光标，完美展示最终 Markdown 报告
                report_container.markdown(full_response)

            except Exception as llm_err:
                st.error(f"❌ 专家技术报告生成中断，请尝试刷新重试。错误原因: {llm_err}")
    else:
        st.info("📢 暂无待处理数据。请在左侧上传一个示功图 CSV 文件，智能体将实时为您诊断。")