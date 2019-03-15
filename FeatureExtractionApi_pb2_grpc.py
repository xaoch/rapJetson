# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc
from grpc.framework.common import cardinality
from grpc.framework.interfaces.face import utilities as face_utilities

import FeatureExtractionApi_pb2 as FeatureExtractionApi__pb2


class FeatureExtractionStub(object):

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.processVideo = channel.stream_unary(
        '/FeatureExtraction/processVideo',
        request_serializer=FeatureExtractionApi__pb2.Image.SerializeToString,
        response_deserializer=FeatureExtractionApi__pb2.Response.FromString,
        )


class FeatureExtractionServicer(object):

  def processVideo(self, request_iterator, context):
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_FeatureExtractionServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'processVideo': grpc.stream_unary_rpc_method_handler(
          servicer.processVideo,
          request_deserializer=FeatureExtractionApi__pb2.Image.FromString,
          response_serializer=FeatureExtractionApi__pb2.Response.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'FeatureExtraction', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))