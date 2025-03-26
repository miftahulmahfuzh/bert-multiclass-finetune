# to clear redis cache storage, run
# python -m redis_compose.flush_redis

import redis
from config import settings

# Initialize Redis client
if settings.REDIS_URL:
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
else:
    redis_client = None

def clear_redis_cache():
    if redis_client:
        redis_client.flushdb()
        print("Redis cache cleared successfully.")
    else:
        print("No Redis client available.")

def clear_namespaced_cache(namespace):
    if redis_client:
        # Get all keys matching the namespace pattern
        keys = redis_client.keys(f"{namespace}:*")
        if keys:
            redis_client.delete(*keys)
            print(f"Cleared {len(keys)} keys from namespace '{namespace}'.")
        else:
            print(f"No keys found for namespace '{namespace}'.")
    else:
        print("No Redis client available.")

if __name__=="__main__":
    # clear_redis_cache()
    clear_namespaced_cache("ai_chatbot_conv")  # Clears only conversation cache
