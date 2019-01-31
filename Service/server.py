import grpc
from concurrent import futures
import time

import all_in_one

from service_spec import all_in_one_pb2
from service_spec import all_in_one_pb2_grpc


class All_In_OneServicer(all_in_one_pb2_grpc.All_In_OneServicer):
    def classify(self, request, context):
        if request.image is None:
            raise InvalidParams("Image is required")
        if request.image_type is None:
            raise InvalidParams("Image type is required")

        temp_responce = all_in_one.predict_image(request.image, request.image_type)

        bounding_box_list = []
        age_list = []
        smile_list = []
        gender_list = []
        for temp_responce_element in temp_responce:
            bounding_box_list.append(all_in_one_pb2.BoundingBox(x=temp_responce_element[0], y=temp_responce_element[1],
                                                                w=temp_responce_element[2], h=temp_responce_element[3]))
            age_list.append(int(temp_responce_element[4]))
            smile_list.append(temp_responce_element[5])
            gender_list.append(temp_responce_element[6])

        responce = all_in_one_pb2.All_In_One_Response(bounding_boxes=bounding_box_list, age=age_list, smile=smile_list,
                                                      gender=gender_list)

        return responce


class Server():
    def __init__(self):
        self.port = '[::]:50051'
        self.server = None

    def start_server(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        all_in_one_pb2_grpc.add_All_In_OneServicer_to_server(All_In_OneServicer(), self.server)
        print('Starting server. Listening on port 50051.')
        self.server.add_insecure_port(self.port)
        self.server.start()

    def stop_server(self):
        self.server.stop(0)
