# simulation/factory.py — SimPy-based digital twin for machines A, B, C

import random
import threading
import simpy
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable


# ── Machine dataclass ──────────────────────────────────────────────────────────

@dataclass
class Machine:
    id: str
    # Sensor state
    air_temperature:    float = 300.0   # Kelvin
    process_temperature: float = 310.0  # Kelvin
    rotational_speed:   float = 1500.0  # rpm
    torque:             float = 40.0    # Nm
    tool_wear:          int   = 0       # minutes
    # Status
    status:             str   = "Running"
    failure_probability: float = 0.0
    # Alert message from LLM (empty string = no active alert)
    alert:              str   = ""
    # Rolling history for dashboard charts
    history:            deque = field(default_factory=lambda: deque(maxlen=100))
    # Step counter for anomaly injection and alert cooldown
    _step:              int   = 0
    _alert_cooldown:    int   = 0

    def snapshot(self) -> dict:
        """Return current sensor readings as a flat dict."""
        return {
            "machine_id":          self.id,
            "timestamp":           datetime.now().isoformat(),
            "air_temperature":     round(self.air_temperature, 2),
            "process_temperature": round(self.process_temperature, 2),
            "rotational_speed":    round(self.rotational_speed, 1),
            "torque":              round(self.torque, 2),
            "tool_wear":           self.tool_wear,
            "failure_probability": round(self.failure_probability, 4),
            "status":              self.status,
            "alert":               self.alert,
        }


# ── Simulation environment ─────────────────────────────────────────────────────

class FactorySimulation:
    """
    Continuous SimPy simulation of three machines.

    On each tick the simulation:
      1. Adds Gaussian noise to sensor values.
      2. Increments tool wear.
      3. Occasionally injects anomaly spikes (~10% chance every 50 steps).
      4. Calls an external predictor callback to update failure_probability.
      5. Updates machine status based on the probability.
      6. Appends a snapshot to the machine's rolling history.
    """

    def __init__(
        self,
        predictor_fn:  Callable[[dict], float] | None = None,
        alert_fn:      Callable[[dict], str]   | None = None,
        time_step:     float = 1.0,
        speed_factor:  float = 1.0,
    ):
        """
        Parameters
        ----------
        predictor_fn : callable(snapshot) -> float
            Returns failure probability for a given sensor snapshot.
        alert_fn : callable(snapshot) -> str
            Returns an LLM alert string (called only when prob >= threshold).
        time_step : float
            Wall-clock seconds between simulation ticks.
        speed_factor : float
            1x / 2x / 5x playback speed multiplier.
        """
        self.predictor_fn  = predictor_fn
        self.alert_fn      = alert_fn
        self.time_step     = time_step
        self.speed_factor  = speed_factor
        self.running       = False
        self._thread: threading.Thread | None = None

        # Initialise machines with slightly different starting conditions
        self.machines: dict[str, Machine] = {
            "A": Machine(id="A", air_temperature=298.5, process_temperature=308.0,
                         rotational_speed=1600.0, torque=38.0, tool_wear=0),
            "B": Machine(id="B", air_temperature=302.0, process_temperature=311.5,
                         rotational_speed=2000.0, torque=55.0, tool_wear=15),
            "C": Machine(id="C", air_temperature=306.0, process_temperature=315.0,
                         rotational_speed=2500.0, torque=70.0, tool_wear=30),
        }

        self.env = simpy.Environment()

    # ── Sensor update helpers ──────────────────────────────────────────────────

    @staticmethod
    def _add_noise(value: float, std: float) -> float:
        return value + random.gauss(0, std)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _inject_anomaly(self, m: Machine) -> None:
        """Spike one random sensor to simulate sudden degradation."""
        spike_type = random.choice(["temp", "speed", "torque"])
        if spike_type == "temp":
            m.air_temperature += random.uniform(5, 15)
        elif spike_type == "speed":
            m.rotational_speed += random.uniform(200, 600)
        else:
            m.torque += random.uniform(10, 30)

    def _update_sensors(self, m: Machine) -> None:
        """Apply noise, increment wear, and optionally inject an anomaly."""
        m.air_temperature    = self._clamp(
            self._add_noise(m.air_temperature, 0.5), 295, 320)
        m.rotational_speed   = self._clamp(
            self._add_noise(m.rotational_speed, 20), 1200, 2800)
        m.torque             = self._clamp(
            self._add_noise(m.torque, 1.5), 20, 80)
        # Process temp tracks air temp with an offset
        m.process_temperature = self._clamp(
            m.air_temperature + random.uniform(8, 12), 300, 330)
        # Tool wear increments by 1–3 minutes each step
        m.tool_wear += random.randint(1, 3)

        # Anomaly injection every ~50 steps with 10% probability
        if m._step % 50 == 0 and m._step > 0 and random.random() < 0.10:
            self._inject_anomaly(m)

    def _update_status(self, m: Machine) -> None:
        """Derive machine status from failure probability."""
        p = m.failure_probability
        if p >= 0.80:
            m.status = "Critical"
        elif p >= 0.40:
            m.status = "Overloaded"
        else:
            m.status = "Running"

    # ── SimPy process per machine ──────────────────────────────────────────────

    def _machine_process(self, env: simpy.Environment, m: Machine):
        while True:
            m._step += 1

            # 1. Evolve sensor readings
            self._update_sensors(m)

            # 2. Get failure probability from the model
            snap = m.snapshot()
            if self.predictor_fn is not None:
                m.failure_probability = self.predictor_fn(snap)
            else:
                # Fallback: simple heuristic so dashboard works without a model
                m.failure_probability = min(
                    1.0,
                    (m.tool_wear / 300) * 0.5 +
                    max(0, (m.torque - 60) / 100) * 0.3 +
                    max(0, (m.rotational_speed - 2500) / 1000) * 0.2,
                )

            # 3. Update status
            self._update_status(m)

            # 4. Trigger LLM alert if above threshold and cooldown has expired
            m._alert_cooldown = max(0, m._alert_cooldown - 1)
            if (
                m.failure_probability >= 0.80
                and m._alert_cooldown == 0
                and self.alert_fn is not None
            ):
                snap = m.snapshot()
                m.alert           = self.alert_fn(snap)
                m._alert_cooldown = 10
            elif m.failure_probability < 0.80:
                m.alert = ""

            # 5. Store snapshot in rolling history
            m.history.append(m.snapshot())

            yield env.timeout(1)   # 1 simulated time unit per tick

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch simulation in a background thread."""
        if self.running:
            return
        self.running = True

        for m in self.machines.values():
            self.env.process(self._machine_process(self.env, m))

        def _run():
            import time
            while self.running:
                self.env.step()
                delay = self.time_step / max(self.speed_factor, 0.1)
                time.sleep(delay)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        print("[factory] Simulation started in background thread.")

    def stop(self) -> None:
        self.running = False
        print("[factory] Simulation stopped.")

    def get_machine(self, machine_id: str) -> Machine:
        return self.machines[machine_id]

    def get_all_snapshots(self) -> dict[str, dict]:
        return {mid: m.snapshot() for mid, m in self.machines.items()}

    def set_speed(self, factor: float) -> None:
        self.speed_factor = factor


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    sim = FactorySimulation(time_step=0.2, speed_factor=1.0)
    sim.start()
    for _ in range(5):
        time.sleep(1)
        snaps = sim.get_all_snapshots()
        for mid, s in snaps.items():
            print(f"  {mid}: wear={s['tool_wear']}  prob={s['failure_probability']:.3f}  status={s['status']}")
    sim.stop()
