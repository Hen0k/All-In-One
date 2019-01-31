import sys

sys.path.insert(0, 'Service/')

from client import ClientTest
from server import *
import unittest
import base64


class TestSuiteGrpc(unittest.TestCase):
    def setUp(self):
        with open('imgs/adele.png', 'rb') as f:
            img = f.read()
            self.image = base64.b64encode(img).decode('utf-8')
            self.image_type = 'RGB'
        self.server = Server()
        self.server.start_server()
        self.client = ClientTest()

    def test_grpc_call(self):
        stub = self.client.open_grpc_channel()
        result = self.client.send_request(stub, self.image, self.image_type)

        expected_result = [[62, 211, 194, 344, 0, 'False', 'Female']]

        bounding_box_list = []
        age_list = []
        smile_list = []
        gender_list = []
        for temp_responce_element in expected_result:
            bounding_box_list.append(all_in_one_pb2.BoundingBox(x=temp_responce_element[0], y=temp_responce_element[1],
                                                                w=temp_responce_element[2], h=temp_responce_element[3]))
            age_list.append(int(temp_responce_element[4]))
            smile_list.append(temp_responce_element[5])
            gender_list.append(temp_responce_element[6])

        expected_responce = all_in_one_pb2.All_In_One_Response(bounding_boxes=bounding_box_list, age=age_list, smile=smile_list,
                                                      gender=gender_list)

        self.assertEqual(result,expected_responce)

    def tearDown(self):
        # self.client.channel.close()
        self.server.stop_server()


if __name__ == '__main__':
    suite = unittest.TestSuite()
    unittest.main()
