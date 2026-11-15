"""Versioned JSON client for physical controller endpoints and logical RL slots."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import ProxyHandler, Request, build_opener

from .settings import Scenario


class ControllerUnavailable(RuntimeError):
    pass


class JointActionError(RuntimeError):
    def __init__(
        self,
        acknowledgments: dict[str, dict[str, Any]],
        failures: dict[str, str],
    ) -> None:
        super().__init__("Joint action failed for {}".format(", ".join(failures)))
        self.acknowledgments = acknowledgments
        self.failures = failures


class ControllerClient:
    def __init__(self, scenario: Scenario, request_timeout: float = 0.6) -> None:
        self.fingerprint = scenario.fingerprint
        self.endpoints = {
            row["id"]: "http://{}:{}".format(row["host"], row["http_port"])
            for row in scenario.data["controllers"]
        }
        self.timeout = request_timeout

    def _attempt(
        self,
        endpoint: str,
        path: str,
        query: dict[str, str | int],
        action: int | None = None,
    ) -> dict[str, Any]:
        url = self.endpoints[endpoint] + path + "?" + urlencode(query)
        headers = {"Content-Type": "application/json"} if action is not None else {}
        data = json.dumps([action]).encode() if action is not None else None
        request = Request(
            url,
            data=data,
            headers=headers,
            method="PUT" if action is not None else "GET",
        )
        # Scenario endpoints are explicit; do not send lab traffic through a proxy.
        with build_opener(ProxyHandler({})).open(
            request, timeout=self.timeout
        ) as response:
            return json.load(response)

    def _call_slot(
        self,
        slot: str,
        path: str,
        query: dict[str, str | int],
        action: int | None = None,
    ) -> dict[str, Any]:
        failures = []
        for endpoint in (slot, *(peer for peer in self.endpoints if peer != slot)):
            try:
                payload = self._attempt(
                    endpoint, path, dict(query, controller_id=slot), action
                )
                if payload["controller_id"] != slot:
                    raise ValueError("Mismatched controller ID in response")
                return payload
            except HTTPError as error:
                if error.code not in (502, 503, 504):
                    raise RuntimeError(
                        f"Controller {endpoint} returned HTTP {error.code}"
                    ) from error
                failures.append(str(error))
            except (OSError, URLError) as error:
                failures.append(str(error))
        raise ControllerUnavailable("{}: {}".format(slot, " | ".join(failures)))

    def fetch_stats(self, slot: str, view: str) -> dict[str, Any]:
        """Return one timestamped snapshot or raise if the slot has no ready owner."""
        sample = self._call_slot(slot, "/network/stats", {"view": view})
        if sample["scenario"] != self.fingerprint or sample["schema_version"] != 1:
            raise ValueError(
                f"Controller {slot} is running a different scenario or schema"
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
        acknowledged = {}
        failures = {slot: "deadline elapsed" for slot in actions}
        while failures and time.monotonic() < deadline:
            pending = tuple(failures)
            fatal = None
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                futures = {
                    slot: executor.submit(
                        self._call_slot,
                        slot,
                        "/model/actions",
                        {"details": 1, "scenario": self.fingerprint},
                        actions[slot],
                    )
                    for slot in pending
                }
                for slot, future in futures.items():
                    try:
                        ack = future.result()
                        if (
                            ack["action"] != actions[slot]
                            or ack["scenario"] != self.fingerprint
                        ):
                            raise ValueError(
                                "Controller acknowledged a different action or scenario"
                            )
                        acknowledged[slot] = ack
                        del failures[slot]
                    except ControllerUnavailable as error:
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
