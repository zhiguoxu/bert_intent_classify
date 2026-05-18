# https://gemini.google.com/share/a3f3e05df007
# 按照上面连接提供的方法部署
#fuser -k -9 10001/tcp

nohup conda run -n bert_intent_classify --no-capture-output \
 uvicorn infer:app --host 0.0.0.0 --port 10001 --workers 4 > infer.log 2>&1 &