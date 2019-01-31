import os
import grpc
import base64
import tempfile
import cv2
import argparse
from inspect import getsourcefile
import os.path
import sys

current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

from service_spec import all_in_one_pb2
from service_spec import all_in_one_pb2_grpc


class ClientTest():
    def __init__(self, port='localhost:50051', image_output='client_out'):
        self.port = port
        self.image_output = image_output

    def open_grpc_channel(self):
        channel = grpc.insecure_channel(self.port)
        stub = all_in_one_pb2_grpc.All_In_OneStub(channel)
        return stub

    def send_request(self, stub, img, image_type='RGB'):
        image_file = all_in_one_pb2.All_In_One_Request(image=img, image_type=image_type)
        response = stub.classify(image_file)
        return response

    def close_channel(self, channel):
        pass


if __name__ == "__main__":
    with open('../imgs/adele.png', 'rb') as f:
        img = f.read()
        image = base64.b64encode(img).decode('utf-8')
    client_test = ClientTest()
    stub = client_test.open_grpc_channel()
    result = client_test.send_request(stub, image)
    print(result)



