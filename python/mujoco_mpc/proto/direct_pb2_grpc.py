# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from . import direct_pb2 as direct__pb2

GRPC_GENERATED_VERSION = '1.73.0'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in direct_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class DirectStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Init = channel.unary_unary(
                '/direct.Direct/Init',
                request_serializer=direct__pb2.InitRequest.SerializeToString,
                response_deserializer=direct__pb2.InitResponse.FromString,
                _registered_method=True)
        self.Data = channel.unary_unary(
                '/direct.Direct/Data',
                request_serializer=direct__pb2.DataRequest.SerializeToString,
                response_deserializer=direct__pb2.DataResponse.FromString,
                _registered_method=True)
        self.Settings = channel.unary_unary(
                '/direct.Direct/Settings',
                request_serializer=direct__pb2.SettingsRequest.SerializeToString,
                response_deserializer=direct__pb2.SettingsResponse.FromString,
                _registered_method=True)
        self.Cost = channel.unary_unary(
                '/direct.Direct/Cost',
                request_serializer=direct__pb2.CostRequest.SerializeToString,
                response_deserializer=direct__pb2.CostResponse.FromString,
                _registered_method=True)
        self.Noise = channel.unary_unary(
                '/direct.Direct/Noise',
                request_serializer=direct__pb2.NoiseRequest.SerializeToString,
                response_deserializer=direct__pb2.NoiseResponse.FromString,
                _registered_method=True)
        self.Reset = channel.unary_unary(
                '/direct.Direct/Reset',
                request_serializer=direct__pb2.ResetRequest.SerializeToString,
                response_deserializer=direct__pb2.ResetResponse.FromString,
                _registered_method=True)
        self.Optimize = channel.unary_unary(
                '/direct.Direct/Optimize',
                request_serializer=direct__pb2.OptimizeRequest.SerializeToString,
                response_deserializer=direct__pb2.OptimizeResponse.FromString,
                _registered_method=True)
        self.Status = channel.unary_unary(
                '/direct.Direct/Status',
                request_serializer=direct__pb2.StatusRequest.SerializeToString,
                response_deserializer=direct__pb2.StatusResponse.FromString,
                _registered_method=True)
        self.SensorInfo = channel.unary_unary(
                '/direct.Direct/SensorInfo',
                request_serializer=direct__pb2.SensorInfoRequest.SerializeToString,
                response_deserializer=direct__pb2.SensorInfoResponse.FromString,
                _registered_method=True)


class DirectServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Init(self, request, context):
        """Initialize Direct
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Data(self, request, context):
        """Set Direct data
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Settings(self, request, context):
        """Direct settings
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Cost(self, request, context):
        """Direct costs
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Noise(self, request, context):
        """Direct noise (process + sensor)
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Reset(self, request, context):
        """Reset Direct
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Optimize(self, request, context):
        """Optimize Direct
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Status(self, request, context):
        """Get Direct status
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SensorInfo(self, request, context):
        """Sensor dimension info
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DirectServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Init': grpc.unary_unary_rpc_method_handler(
                    servicer.Init,
                    request_deserializer=direct__pb2.InitRequest.FromString,
                    response_serializer=direct__pb2.InitResponse.SerializeToString,
            ),
            'Data': grpc.unary_unary_rpc_method_handler(
                    servicer.Data,
                    request_deserializer=direct__pb2.DataRequest.FromString,
                    response_serializer=direct__pb2.DataResponse.SerializeToString,
            ),
            'Settings': grpc.unary_unary_rpc_method_handler(
                    servicer.Settings,
                    request_deserializer=direct__pb2.SettingsRequest.FromString,
                    response_serializer=direct__pb2.SettingsResponse.SerializeToString,
            ),
            'Cost': grpc.unary_unary_rpc_method_handler(
                    servicer.Cost,
                    request_deserializer=direct__pb2.CostRequest.FromString,
                    response_serializer=direct__pb2.CostResponse.SerializeToString,
            ),
            'Noise': grpc.unary_unary_rpc_method_handler(
                    servicer.Noise,
                    request_deserializer=direct__pb2.NoiseRequest.FromString,
                    response_serializer=direct__pb2.NoiseResponse.SerializeToString,
            ),
            'Reset': grpc.unary_unary_rpc_method_handler(
                    servicer.Reset,
                    request_deserializer=direct__pb2.ResetRequest.FromString,
                    response_serializer=direct__pb2.ResetResponse.SerializeToString,
            ),
            'Optimize': grpc.unary_unary_rpc_method_handler(
                    servicer.Optimize,
                    request_deserializer=direct__pb2.OptimizeRequest.FromString,
                    response_serializer=direct__pb2.OptimizeResponse.SerializeToString,
            ),
            'Status': grpc.unary_unary_rpc_method_handler(
                    servicer.Status,
                    request_deserializer=direct__pb2.StatusRequest.FromString,
                    response_serializer=direct__pb2.StatusResponse.SerializeToString,
            ),
            'SensorInfo': grpc.unary_unary_rpc_method_handler(
                    servicer.SensorInfo,
                    request_deserializer=direct__pb2.SensorInfoRequest.FromString,
                    response_serializer=direct__pb2.SensorInfoResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'direct.Direct', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('direct.Direct', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class Direct(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Init(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/direct.Direct/Init',
            direct__pb2.InitRequest.SerializeToString,
            direct__pb2.InitResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Data(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/direct.Direct/Data',
            direct__pb2.DataRequest.SerializeToString,
            direct__pb2.DataResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Settings(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/direct.Direct/Settings',
            direct__pb2.SettingsRequest.SerializeToString,
            direct__pb2.SettingsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Cost(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/direct.Direct/Cost',
            direct__pb2.CostRequest.SerializeToString,
            direct__pb2.CostResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Noise(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/direct.Direct/Noise',
            direct__pb2.NoiseRequest.SerializeToString,
            direct__pb2.NoiseResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Reset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/direct.Direct/Reset',
            direct__pb2.ResetRequest.SerializeToString,
            direct__pb2.ResetResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Optimize(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/direct.Direct/Optimize',
            direct__pb2.OptimizeRequest.SerializeToString,
            direct__pb2.OptimizeResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Status(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/direct.Direct/Status',
            direct__pb2.StatusRequest.SerializeToString,
            direct__pb2.StatusResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def SensorInfo(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/direct.Direct/SensorInfo',
            direct__pb2.SensorInfoRequest.SerializeToString,
            direct__pb2.SensorInfoResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
