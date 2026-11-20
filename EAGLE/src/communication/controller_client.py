"""Versioned JSON client for physical controller endpoints and logical RL slots."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from EAGLE.src.network import Network
from network_schema import valid_support_action


class ControllerUnavailableError(RuntimeError):
    """No ready physical controller could serve a logical slot."""


class JointActionError(RuntimeError):
    """A joint action failed, retaining successful acknowledgments and slot failures."""

    def __init__(
        self,
        acknowledgments: dict[str, dict[str, Any]],
        failures: dict[str, str],
    ) -> None:
        super().__init__("Joint action failed for {}".format(", ".join(failures)))
        self.acknowledgments = acknowledgments
        self.failures = failures


class ControllerClient:
    """Address logical controller slots through their primary endpoint or live peers."""

    def __init__(self, network: Network, request_timeout: float = 0.6) -> None:
        self.controllers = {row["id"]: row for row in network.data["controllers"]}
        self.fingerprint = network.fingerprint
        self.endpoints = {
            row["id"]: "http://{}:{}".format(row["host"], row["http_port"])
            for row in network.data["controllers"]
        }
        self.timeout = request_timeout

    def _attempt(
        self,
        endpoint: str,
        path: str,
        query: dict[str, str | int],
        action: int | None = None,
    ) -> dict[str, Any]:
        """Issue one proxy-free JSON request to a physical controller endpoint."""
        url = self.endpoints[endpoint] + path + "?" + urlencode(query)
        headers = {"Content-Type": "application/json"} if action is not None else {}
        data = json.dumps([action]).encode() if action is not None else None
        request = Request(  # noqa: S310
            url,
            data=data,
            headers=headers,
            method="PUT" if action is not None else "GET",
        )
        # Network endpoints are explicit; do not send lab traffic through a proxy.
        with build_opener(ProxyHandler({})).open(
            request,
            timeout=self.timeout,
        ) as response:
            return json.load(response)

    def _call_slot(
        self,
        slot: str,
        path: str,
        query: dict[str, str | int],
        action: int | None = None,
    ) -> dict[str, Any]:
        """Try a slot primary and then peers, retrying only unavailable endpoints."""
        failures = []
        for endpoint in (slot, *(peer for peer in self.endpoints if peer != slot)):
            try:
                payload = self._attempt(
                    endpoint,
                    path,
                    dict(query, controller_id=slot),
                    action,
                )
                if payload["controller_id"] != slot:
                    raise ValueError("Mismatched controller ID in response")
                return payload  # noqa: TRY300
            except HTTPError as error:
                if error.code not in (502, 503, 504):
                    raise RuntimeError(
                        f"Controller {endpoint} returned HTTP {error.code}",
                    ) from error
                failures.append(str(error))
            except (OSError, URLError) as error:
                failures.append(str(error))
        msg = "{}: {}".format(slot, " | ".join(failures))
        raise ControllerUnavailableError(msg)

    def fetch_stats(self, slot: str) -> dict[str, Any]:
        """Return one timestamped snapshot or raise if the slot has no ready owner."""
        sample = self._call_slot(slot, "/network/stats", {})
        if sample["network"] != self.fingerprint or sample["schema_version"] != 1:
            raise ValueError(
                f"Controller {slot} is running a different network or schema",
            )
        if sample["sample_duration_seconds"] <= 0:
            raise ValueError(f"Controller {slot} reported an invalid poll window")
        return sample

    def apply_actions(
        self,
        actions: dict[str, int],
        deadline: float,
    ) -> dict[str, dict[str, Any]]:
        """Send idempotent action requests and retain all successful acknowledgments."""
        if any(
            slot not in self.controllers
            or not valid_support_action(self.controllers[slot], action)
            for slot, action in actions.items()
        ):
            raise ValueError(
                "Actions must be valid support-switch masks for configured controllers"
            )
        acknowledged = {}
        failures = dict.fromkeys(actions, "deadline elapsed")
        while failures and time.monotonic() < deadline:
            pending = tuple(failures)
            fatal = None
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = {
                    slot: executor.submit(
                        self._call_slot,
                        slot,
                        "/model/actions",
                        {"network": self.fingerprint},
                        actions[slot],
                    )
                    for slot in pending
                }
                for slot, future in futures.items():
                    try:
                        ack = future.result()
                        if (
                            ack["action"] != actions[slot]
                            or ack["network"] != self.fingerprint
                        ):
                            raise ValueError(  # noqa: TRY301
                                "Controller acknowledged a different action or network",
                            )
                        acknowledged[slot] = ack
                        del failures[slot]
                    except ControllerUnavailableError as error:
                        failures[slot] = str(error)
                    except (ValueError, KeyError, RuntimeError) as error:
                        failures[slot] = str(error)
                        fatal = error
            if fatal is not None:
                raise JointActionError(acknowledged, failures) from fatal
            if failures:
                time.sleep(0.1)
        if failures:
            raise JointActionError(acknowledged, failures)
        return acknowledged
