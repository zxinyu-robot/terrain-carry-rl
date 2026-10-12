# TerrainCarry-RL

Tightly coupled whole-body reinforcement learning for a quadruped manipulator
carrying objects across uneven terrain.

The project focuses on three engineering problems:

- transfer a pretrained locomotion/manipulation policy to a contact-rich carry task;
- distill a privileged simulation teacher into a deployable observation-only student;
- make the learned policy auditable for sim-to-sim and eventual sim-to-real transfer.

The initial implementation is a research scaffold. It does not claim physical
robot deployment or reproduce third-party Isaac Lab assets.
