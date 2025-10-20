#!/bin/bash
# Monitor training progress

echo "==================================================================="
echo "Training Monitor - $(date)"
echo "==================================================================="
echo ""

# Check if process is running
PID=$(ps aux | grep "train_deep_learning.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$PID" ]; then
    echo "❌ Training process NOT running"
    echo ""
    echo "Checking for results..."
    if [ -f "../results/deep_learning_results.json" ]; then
        echo "✓ Results file exists!"
        cat ../results/deep_learning_results.json
    else
        echo "❌ No results file found"
    fi
else
    echo "✓ Training process RUNNING (PID: $PID)"
    echo ""
    
    # Show process info
    echo "Process Details:"
    ps aux | grep "$PID" | grep -v grep | awk '{printf "  CPU: %s%%\n  Memory: %s MB\n  Time: %s\n", $3, int($6/1024), $10}'
    echo ""
    
    # Check log file
    if [ -f "/tmp/training.log" ]; then
        LOG_SIZE=$(wc -l < /tmp/training.log)
        echo "Log file: $LOG_SIZE lines"
        if [ "$LOG_SIZE" -gt 0 ]; then
            echo ""
            echo "Recent log output:"
            echo "-------------------------------------------------------------------"
            tail -30 /tmp/training.log
            echo "-------------------------------------------------------------------"
        else
            echo "  (Log file empty - data loading phase)"
        fi
    else
        echo "No log file at /tmp/training.log"
    fi
    
    echo ""
    
    # Check for model checkpoints
    echo "Checking for saved models..."
    if [ -f "../results/models/lstm_best.pt" ]; then
        LSTM_SIZE=$(ls -lh ../results/models/lstm_best.pt | awk '{print $5}')
        echo "  ✓ LSTM model saved ($LSTM_SIZE)"
    else
        echo "  ⏳ LSTM model not yet saved"
    fi
    
    if [ -f "../results/models/transformer_best.pt" ]; then
        TRANS_SIZE=$(ls -lh ../results/models/transformer_best.pt | awk '{print $5}')
        echo "  ✓ Transformer model saved ($TRANS_SIZE)"
    else
        echo "  ⏳ Transformer model not yet saved"
    fi
fi

echo ""
echo "==================================================================="
echo "To monitor continuously: watch -n 30 ./monitor_training.sh"
echo "==================================================================="
