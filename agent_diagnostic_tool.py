import os
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from model_structure import Compressor1DCNN

# 设置绘图支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CompressorAgentTool:
    def __init__(self, model_path="compressor_cnn_model.pth",
                 encoder_path="label_encoder.pkl",
                 scaler_path="data_scaler.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.exists(encoder_path) and os.path.exists(scaler_path):
            self.label_encoder = joblib.load(encoder_path)
            self.scaler = joblib.load(scaler_path)
            num_classes = len(self.label_encoder.classes_)
        else:
            raise FileNotFoundError("未找到 pkl 配置文件，请确保之前的训练脚本成功运行！")

        self.model = Compressor1DCNN(num_classes=num_classes)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("【智能体工具】往复压缩机底层神经网络大脑加载成功！")
        else:
            raise FileNotFoundError(f"找不到模型权重文件: {model_path}")

    def _calculate_indicator_work(self, volume, pressure):
        """
        核心升级：基于格林公式/数值积分计算封闭示功图的面积（单圈指示功）
        """
        # 使用鞋带公式（Shoelace formula）或沿路径的梯形积分计算封闭曲线面积
        return float(0.5 * np.abs(np.dot(volume, np.roll(pressure, 1)) - np.dot(pressure, np.roll(volume, 1))))

    def _generate_pv_plot(self, volume, pressure, label_name, save_path="diagnostic_pv.png"):
        """
        核心升级：自动绘制当前样本的示功图并保存，供现场工程师查看
        """
        plt.figure(figsize=(6, 4.5))
        plt.plot(volume, pressure, 'r-', linewidth=2, label=f"当前分析曲线 ({label_name})")
        plt.title("智能体实时诊断 - 示功图 (p-V 图)", fontsize=12, fontweight='bold')
        plt.xlabel("容积 Volume (V)", fontsize=10)
        plt.ylabel("压力 Pressure (P)", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()  # 释放内存
        return os.path.abspath(save_path)

    def diagnose_csv(self, csv_path):
        if not os.path.exists(csv_path):
            return {"status": "error", "message": f"找不到指定的文件: {csv_path}"}

        try:
            # 1. 读取前两列
            df = pd.read_csv(csv_path, header=None, usecols=[0, 1])
            if df.isna().sum().sum() > 0:
                df = df.ffill().bfill()

            volume = df.iloc[:, 0].values
            pressure = df.iloc[:, 1].values

            # 2. 提取定量热力学指标
            max_p = float(np.max(pressure))
            min_p = float(np.min(pressure))
            indicator_work = self._calculate_indicator_work(volume, pressure)

            # 3. 神经网络推理
            raw_feature = df.values.flatten().reshape(1, -1)
            if raw_feature.shape[1] != self.scaler.n_features_in_:
                return {
                    "status": "error",
                    "message": f"输入数据的维度为 {raw_feature.shape[1]}，与模型要求的 {self.scaler.n_features_in_} 维度不符。"
                }

            scaled_feature = self.scaler.transform(raw_feature)
            tensor_feature = torch.tensor(scaled_feature, dtype=torch.float32).unsqueeze(1).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_feature)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            pred_class_idx = int(np.argmax(probabilities))
            pred_label = self.label_encoder.inverse_transform([pred_class_idx])[0]
            confidence = float(probabilities[pred_class_idx])

            # 4. 核心升级：自动联动绘图
            img_path = self._generate_pv_plot(volume, pressure, pred_label)

            # 5. 返回丰富好用的结构化数据字典
            return {
                "status": "success",
                "diagnosis": pred_label,
                "confidence": f"{confidence * 100:.2f}%",
                "saved_plot_path": img_path,
                "metrics": {
                    "maximum_pressure": round(max_p, 2),
                    "minimum_pressure": round(min_p, 2),
                    "indicator_work_area": round(indicator_work, 2)
                }
            }

        except Exception as e:
            return {"status": "error", "message": f"诊断失败，错误原因: {str(e)}"}


if __name__ == "__main__":
    tool = CompressorAgentTool()
    test_folder = r"E:\课题数据\示功图\排气阀泄漏"
    if os.path.exists(test_folder):
        files = [f for f in os.listdir(test_folder) if f.endswith('.csv')]
        if files:
            sample_path = os.path.join(test_folder, files[0])
            res = tool.diagnose_csv(sample_path)
            print("\n[功能升级测试] 底层工具返回的高级结构化结果:")
            import json

            print(json.dumps(res, ensure_ascii=False, indent=4))