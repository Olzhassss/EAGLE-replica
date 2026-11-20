"""HTTP endpoints for network state, telemetry, and acknowledged slot actions."""

import json
import time
from typing import Any, Dict

from network_schema import valid_support_action
from webob import Request, Response

from ryu.app.wsgi import ControllerBase, route


def _response(data: object, status: int = 200) -> Response:
    return Response(
        status=status,
        content_type="application/json",
        body=json.dumps(data).encode(),
    )


class EagleApi(ControllerBase):
    """Expose network state, fresh slot telemetry, and acknowledged policy actions over HTTP."""

    def __init__(
        self,
        request: Request,
        link: Any,
        data: Dict[str, Any],
        **config,
    ) -> None:
        super().__init__(request, link, data, **config)
        self.app = data["app"]

    @route("eagle", "/network/state", methods=["GET"])
    def get_state(self, _: Request, **__) -> Response:
        with self.app.lock:
            return _response(self.app.presence())

    @route("eagle", "/network/stats", methods=["GET"])
    def get_stats(self, request: Request, **_) -> Response:
        """Serve a slot snapshot or report that its owner is waiting for coherent telemetry."""
        slot = request.GET.get("controller_id", self.app.controller_id)
        if slot not in self.app.network.controllers:
            return _response({"error": "Unknown controller slot"}, 400)
        if request.GET.get("view", "active") != "active":
            return _response({"error": "Only active telemetry is supported"}, 400)
        with self.app.lock:
            snapshot = self.app.stats(slot)
            if snapshot is None:
                return _response(
                    {
                        "error": self.app.route_error
                        or "Waiting for ownership and a complete sample",
                        "owner": self.app.topology.owners.get(slot),
                    },
                    503,
                )
            return _response(snapshot)

    @route("eagle", "/model/actions", methods=["PUT"])
    def put_actions(self, request: Request, **_) -> Response:
        """Validate a support-switch bitmask for an owned slot and acknowledge its revision."""
        slot = request.GET.get("controller_id", self.app.controller_id)
        if (
            request.GET.get("network", self.app.network.fingerprint)
            != self.app.network.fingerprint
        ):
            return _response({"error": "Network fingerprint mismatch"}, 409)
        try:
            payload = request.json
        except ValueError:
            payload = None
        if slot not in self.app.network.controllers:
            return _response({"error": "Unknown controller slot"}, 400)
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or not valid_support_action(self.app.network.controllers[slot], payload[0])
        ):
            maximum = (
                1 << len(self.app.network.controllers[slot]["support_switches"])
            ) - 1
            return _response(
                {"error": "Expected [mask] with an integer from 0 to {}".format(maximum)},
                400,
            )
        with self.app.lock:
            if self.app.topology.owners.get(slot) != self.app.controller_id:
                return _response(
                    {
                        "error": "Controller does not own this slot",
                        "owner": self.app.topology.owners.get(slot),
                    },
                    503,
                )
            current = self.app.topology.resources[slot]
            if current["action"] != payload[0]:
                revision = [
                    max(time.time_ns(), current["revision"][0] + 1),
                    self.app.controller_id,
                ]
                self.app.topology.set_support_action(slot, payload[0], revision)
                self.app.telemetry.reset()
                self.app.sequence += 1
            update = self.app.topology.resources[slot]
            accepted_at = time.time()
            return _response(
                {
                    "controller_id": slot,
                    "served_by": self.app.controller_id,
                    "action": update["action"],
                    "network": self.app.network.fingerprint,
                    "revision": update["revision"],
                    "accepted_at": accepted_at,
                    "changed": current != update,
                },
            )
