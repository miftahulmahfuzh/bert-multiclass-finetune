curl -X POST http://10.183.0.2:7860/gradio_api/call/chat -s -H "Content-Type: application/json" -d '{
  "data": [
							"Hello!!"
]}' \
  | awk -F'"' '{ print $4}'  \
  | read EVENT_ID; curl -N http://10.183.0.2:7860/gradio_api/call/chat/$EVENT_ID
