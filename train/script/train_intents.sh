nohup /home/ubuntu/miniconda3/bin/conda run -n intent_classify --no-capture-output python train/train.py intents > "train_intents.log" 2>&1 &
