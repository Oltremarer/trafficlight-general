# Implementation notes

This is the single holding area for local, non-architectural follow-up items. Architectural or data-contract defects are fixed immediately instead of being recorded here.

- The initial metric collector derives throughput from disappearance of previously active vehicle IDs. This is valid for completed flows but does not distinguish a vehicle leaving the network from a simulator-specific removal event.
- QDSE, regional rewards, and diffusion policies remain deferred. The shared DQN is the initial model-free learned baseline.
- The package currently targets CityFlow 0.1 on Linux. Native macOS ARM execution is not part of the supported runtime.
- Time-limit truncation is currently treated as terminal by the DQN TD target, matching a fixed-duration finite-horizon experiment. A continuing-task trainer should expose a separate bootstrap-on-truncation protocol instead of silently changing this behavior.
- CityFlow seeds do not create meaningful variance for a fixed deterministic flow and deterministic policy; controller evaluation variance requires stochastic demand or independently generated flows.
- The initial graph World Model is deterministic and has no ensemble or calibrated uncertainty. It should not be used for unconstrained long rollouts; the default planner stays at horizon 3 and searches only local mutations around MaxPressure.
- The candidate planner does not yet impose explicit safety or minimum-green constraints beyond the environment-owned phase validity and yellow/all-red timing contract.
