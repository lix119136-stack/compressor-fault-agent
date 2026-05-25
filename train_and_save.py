import os
import torch
import numpy as np
import pandas as pd
import joblib
# 引入我们训练脚本里定义的网络结构
from train_and_save import Compressor1DCNN


class CompressorAgentTool:
    def __init__(self, model_path="compressor_cnn_model.pth",
                 encoder_path="label_encoder.pkl",
                 scaler_path="data_scaler.pkl"):
        """
        初始化智能体工具：自动加载训练好的神经网络和配套转换器
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. 加载标签映射器与归一化工具
        if os.path.exists(encoder_path) and os.path.exists(scaler_path):
            self.label_encoder = joblib.load(encoder_path)
            self.scaler = joblib.load(scaler_path)
            num_classes = len(self.label_encoder.classes_)
        else:
            raise FileNotFoundError("未找到 pkl 配置文件，请确保 train_and_save.py 成功运行！")

        # 2. 加载训练好的 PyTorch 模型权重
        self.model = Compressor1DCNN(num_classes=num_classes)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()  # 切换为测试/推理模式
            print("【智能体工具】往复压缩机底层神经网络大脑加载成功！")
        else:
            raise FileNotFoundError(f"找不到模型权重文件: {model_path}")

    def diagnose_csv(self, csv_path):
        """
        输入一个未知的本地 CSV 路径，输出诊断结果、置信度以及热力学特征提取
        """
        if not os.path.exists(csv_path):
            return {"status": "error", "message": f"找不到指定的文件: {csv_path}"}

        try:
            # 1. 精准提取前两列
            df = pd.read_csv(csv_path, header=None, usecols=[0, 1])
            if df.isna().sum().sum() > 0:
                df = df.ffill().bfill()

            # 提取热力学特征，供大模型进行“机理分析”使用
            volume = df.iloc[:, 0].values
            pressure = df.iloc[:, 1].values

            max_p = float(np.max(pressure))
            min_p = float(np.min(pressure))

            # 2. 转换为模型输入的特征向量并归一化
            raw_feature = df.values.flatten().reshape(1, -1)

            # 安全防线：确保输入的行数和训练时一致（722维）
            if raw_feature.shape[1] != self.scaler.n_features_in_:
                return {
                    "status": "error",
                    "message": f"输入数据的维度为 {raw_feature.shape[1]}，与模型要求的 {self.scaler.n_features_in_} 维度不符。"
                }

            scaled_feature = self.scaler.transform(raw_feature)

            # 3. 送入 1D-CNN 进行神经网络推理
            tensor_feature = torch.tensor(scaled_feature, dtype=torch.float32).unsqueeze(1).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor_feature)
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            # 4. 获取概率最高的类别
            pred_class_idx = int(np.argmax(probabilities))
            pred_label = self.label_encoder.inverse_transform([pred_class_idx])[0]
            confidence = float(probabilities[pred_class_idx])

            # 5. 返回结构化字典，供 LLM 智能体理解
            return {
                "status": "success",
                "diagnosis": pred_label,
                "confidence": f"{confidence * 100:.2f}%",
                "features": {
                    "maximum_pressure": round(max_p, 2),
                    "minimum_pressure": round(min_p, 2)
                }
            }

        except Exception as e:
            return {"status": "error", "message": f"诊断失败，错误原因: {str(e)}"}


# =====================================================================
# 局部测试逻辑：模拟智能体调用工具
# =====================================================================
if __name__ == "__main__":
    tool = CompressorAgentTool()

    # 随机挑一个“吸气阀泄漏”文件夹下的文件来做盲测验证
    test_file = r"E:\课题数据\示功图\吸气阀泄漏"
    if os.path.exists(test_file):
        files = [f for f in os.listdir(test_file) if f.endswith('.csv')]
        if files:
            sample_path = os.path.join(test_file, files[5])  # 拿第 6 个样本做测试
            print(f"\n[测试调用] 正在把未知文件送入智能体工具接口:\n-> {sample_path}")

            result = tool.diagnose_csv(sample_path)
            print("\n[工具返回的结构化 JSON 结果]:")
            import json

            print(json.dumps(result, ensure_ascii=False, indent=4))