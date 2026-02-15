
import redis

class RedisProvider:
    def __init__(self, host="localhost", port=6379, db=0, password=None):
        self.config = {"host": host, "port": port, "db": db, "password": password}
        self._pool = None

    def get_client(self):
        if self._pool is None:
            self._pool = redis.ConnectionPool(
                host=self.config["host"],
                port=self.config["port"],
                db=self.config["db"],
                password=self.config["password"],
                decode_responses=False 
            )
        return redis.Redis(connection_pool=self._pool)