rm -rf ../models/bert_onnx
conda run -n bert_intent_classify --no-capture-output \
 optimum-cli export onnx \
  --model ../output/intent_classify_20260521-202839 \
  --optimize O3 \
  --task text-classification \
  ../models/bert_onnx/