#!/bin/bash
# 同时训练LSTM和Transformer

cd /Users/yuchenzhou/Documents/duke/compsci526/final_proj/git_proj/yuchen_testing/yuchen_testing_10.19/src/models

# 激活conda环境
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate readmit_pred

echo "======================================================================"
echo "启动并行训练"
echo "======================================================================"
echo ""

# 启动LSTM训练
echo "启动LSTM训练..."
nohup python train_lstm_fixed.py > /tmp/lstm_training_final.log 2>&1 &
LSTM_PID=$!
echo "  LSTM PID: $LSTM_PID"

# 等待2秒
sleep 2

# 启动Transformer训练
echo "启动Transformer训练..."
nohup python train_transformer.py > /tmp/transformer_training_final.log 2>&1 &
TRANSFORMER_PID=$!
echo "  Transformer PID: $TRANSFORMER_PID"

echo ""
echo "======================================================================"
echo "两个模型正在并行训练！"
echo "======================================================================"
echo ""
echo "查看进度："
echo "  LSTM: tail -f /tmp/lstm_training_final.log"
echo "  Transformer: tail -f /tmp/transformer_training_final.log"
echo ""
echo "估计时间："
echo "  LSTM: ~50分钟 (25 epochs)"
echo "  Transformer: ~2-3小时 (25 epochs)"
echo ""
echo "检查进程："
echo "  ps aux | grep 'train_lstm_fixed\\|train_transformer' | grep -v grep"
echo ""

# 等待几秒后显示初始输出
sleep 5
echo "======================================================================"
echo "LSTM 初始输出："
echo "======================================================================"
tail -15 /tmp/lstm_training_final.log
echo ""
echo "======================================================================"
echo "Transformer 初始输出："
echo "======================================================================"
tail -15 /tmp/transformer_training_final.log
