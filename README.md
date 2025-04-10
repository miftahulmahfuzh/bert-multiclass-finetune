project path in gcp vm (10.183.0.2):
/home/devmiftahul/nlp/faq_chatbot/news_mcp_server2

use 'mcp' virtual environment like so:
source /home/devmiftahul/.pyenv/versions/mcp/bin/activate

for sse route (weather_server.py), we need to run the mcp server separately like so:
python weather_server.py

then, we can run the client like so:
python multiple_mpc_client
