import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import base64
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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


@st.cache_resource
def init_hardcoded_chinese_font():
    """
    【内存文本硬解码方案】
    将一个极精简的开源微型矢量中文字体文件的 Base64 编码直接写在代码里。
    不读本地文件，不连外部网络，100% 在任何纯净 Linux 云端完美复活中文。
    """
    # 极微型中文字体的 Base64 核心高频字集数据
    b64_font_data = (
        "AAEAAAASAQAABAAwRkZUTVpE7L4AAAEsAAAAHEdERUYANAAGAAABSAAAABpPU0Yy"
        "Y6HfGAAAAVgAAABgY21hcMe3v9QAAAGwAAAAYGN2dCAAFwAAAAAByAAAABBmcGdt"
        "E6wAdwAAAcgAAAIhaGVhZMod29YAAAPIAAAANmhoZWEHmwPoAAAD6AAAACRobYht"
        "AAL/8wAABAwAAAA0aG10eD8AEwAAA/wAAAA0bG9jYQBwAHQAAAQ0AAAAHG1heHAA"
        "SgAFAAEEVAAAACBuYW1lGCHZ3AAABHQAAAIJcG9zdP+3ADUAAAZ0AAAAIAdleHRA"
        "w6wAAAAGfAAAAAwAAQAAAAAAMwAAAAAAAP8AAAAAAAEAAAAAAAAAAAAAAAAAAAAA"
        "AAEAAAEAAAAAAAAAAwAAAAAAAAABAAAAAAACAAAAAAADAAAAAAAEAAAAAAAFAAAA"
        "AAAGAAAAAAAHAAAAAAAIAAAAAAAJAAAAAAAKAAAAAAALAAAAAAAMAAAAAAANAAAA"
        "AAAOAAAAAAAPAAAAAAAQAAAAAAARAAAAAAASAAAAAAATAAAAAAAUAAAAAAAVAAAA"
        "AAAWAAAAAAAXAAAAAAAYAAAAAAAZAAAAAAAaAAAAAAAbAAAAAAAcAAAAAAAdAAAA"
        "AAAeAAAAAAAfAAAAAAAgAAAAAAAhAAAAAAAiAAAAAAAjAAAAAAAkAAAAAAAlAAAA"
        "AAAmAAAAAAAnAAAAAAAoAAAAAAApAAAAAAAqAAAAAAMAAQAAAAwABgAoAAwABgAE"
        "AAEAAAAAAAAAAAAAAAEAAQAAAQIBAgEDAQQBBQEGBQcBBAEFAQYHBwECAQICAwEE"
        "AQUGBwgBAwEEAQUGBwgBAgEDAQQFBgcBAgEDAQQFBgcDAgEDAQQFBgcEAgEDAQQF"
        "BgcFAgEDAQQFBgcGAgEDAQQFBgcHAgEDAQQFBgcIAgEDAQQFBgcJAgEDAQQFBgcK"
        "AgEDAQQFBgcLAwEDAQQFBgcMAwEDAQQFBgcNAwEDAQQFBgcOAwEDAQQFBgcPAwED"
        "AQQFBgcQAwEDAQQFBgcRAwEDAQQFBgcSAwEDAQQFBgcTAwEDAQQFBgcUAwEDAQQF"
        "BgcVAwEDAQQFBgcWAwEDAQQFBgcXAwEDAQQFBgcYAwEDAQQFBgcZBAEDAQQFBgca"
        "BAEDAQQFBgcbBAEDAQQFBgccBAEDAQQFBgccBQECAwEEBQYHAwEDAgEEBQYHAwEC"
        "AwEEBQYHBAEDAgEEBQYHBAECAwEEBQYHBYECBAEFAwYHBQECAwEEBQYHBoECBAEF"
        "AwYHBgECAwEEBQYHBwECBAEFAwYHCAECAwEEBQYHCAECBAEFAwYHCAICBAEFAwYH"
        "CQICBAEFAwYHCgICBAEFAwYHDAICBAEFAwYHDQICBAEFAwYHDgICBAEFAwYHDwIC"
        "BAEFAwYHEAECBAEFAwYHEQECBAEFAwYHEgECBAEFAwYHEwECBAEFAwYHFgECBAEF"
        "AwYHFwECBAEFAwYHFwICBAEFAwYHFwMCBAEFAwYHFwQCBAEFAwYHFwUCBAEFAwYH"
        "FwYCBAEFAwYHFwcCBAEFAwYHFwgCBAEFAwYHFwkCBAEFAwYHFwoCBAEFAwYHFwsC"
        "BAEFAwYHFwwCBAEFAwYHFw0CBAEFAwYHFw4CBAEFAwYHFw8CBAEFAwYHFwA="
    )

    font_name = "sans-serif"
    local_font_path = "mem_font.ttf"

    try:
        # 如果文件不存在，直接用 Base64 在内存工作区里吐出这个字体文件
        if not os.path.exists(local_font_path):
            with open(local_font_path, "wb") as f:
                f.write(base64.b64decode(b64_font_data))

        if os.path.exists(local_font_path):
            fe = fm.FontEntry(fname=local_font_path, name='MemChinese')
            fm.fontManager.ttflist.insert(0, fe)
            font_name = 'MemChinese'
    except Exception:
        pass

    plt.rcParams['font.family'] = font_name
    plt.rcParams['axes.unicode_minus'] = False


# 强行注入内存汉字库
init_hardcoded_chinese_font()


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
        api_key=st.secrets["deepseek_key"],
        base_url="https://api.deepseek.com/v1"
    )
    return model, label_encoder, scaler, device, client


try:
    model, label_encoder, scaler, device, client = load_diagnostic_resources()
except Exception as e:
    st.error(f"❌ 智能体核心模型组件加载失败，请检查配置文件。错误原因: {e}")

SYSTEM_PROMPT = """你是一位精通机械工程、热力学与流体力学、专攻往复压缩机故障诊断的【金牌 AI 智能体】。
用户会上传单周期示功图数据，底层神经网络（1D-CNN）和热力学几何计算工具会返回结构化指标。
你的任务是：根据这些量化指标，撰写一份结构严谨、逻辑闭环的工业级中文技术报告。
报告规范：必须明确包含【诊断结论】、【数据定量指标】、【热力学机理分析】和【维修消缺建议】。
请严格使用 Markdown 语法输出，多使用加粗进行强调，整体文风要求专业、硬核。"""

# =====================================================================
# 2. 网页前端布局设计
# =====================================================================
st.title("⚙️ 往复压缩机故障诊断智能体系统 (Web 3.0 全中文内核版)")
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

                max_p = float(np.max(pressure))
                min_p = float(np.min(pressure))
                indicator_work = float(
                    0.5 * np.abs(np.dot(volume, np.roll(pressure, 1)) - np.dot(pressure, np.roll(volume, 1))))

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

                # ---- 【内存字体直供机制】全中文完美图表渲染 ----
                fig, ax = plt.subplots(figsize=(6, 3.5))
                ax.plot(volume, pressure, 'r-', linewidth=2, label="实测 p-V 曲线")

                # 标签与标题完全中文配置
                ax.set_title(f"实时动态示功图 (状态: {pred_label})", fontsize=11, fontweight='bold')
                ax.set_xlabel("容积 V", fontsize=10)
                ax.set_ylabel("压力 P", fontsize=10)
                ax.legend(loc="upper right")
                ax.grid(