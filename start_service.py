import sys

sys.path.insert(0, 'Service/')

from server import *

server = Server()
server.start_server()

try:
    while True:
        ONE_DAY = 60 * 60 * 24
        time.sleep(ONE_DAY)
except KeyboardInterrupt:
    server.stop_server()
