#!/usr/bin/env python3
import pyarrow.flight as flight

class MyFlightServer(flight.FlightServerBase):
    def list_flights(self, context, criteria):
        info = flight.FlightInfo(...)
        yield info

import os

if os.environ.get('CLIENT', None):
    client = flight.connect("grpc://0.0.0.0:53403")
    print(list(client.list_flights()))
    
else:

    # Listen to all interfaces on a free port
    server = MyFlightServer("grpc://0.0.0.0:0")
    
    print("Server listening on port", server.port)
    server.serve()
