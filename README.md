# Drone Simulation in Webots using Tellopy API

This project simulates the behavior of a drone in Webots using an API designed to replicate the control functions of a real drone using the [Tellopy](https://github.com/hanyazou/TelloPy) library. The simulation focuses on replicating the movement controls of the drone as closely as possible, although **there is no guarantee that the speeds between the real and simulated drones are identical**.

## Description

To ensure that the simulated drone in Webots behaves similarly to the real drone controlled via the `tellopy` API, a specific API was developed to mimic the main movement controls. The control functions provided by the `tellopy` API are as follows:

- **set_throttle**: Controls vertical movement (up/down).
- **set_yaw**: Controls rotation around the vertical axis (left/right turn).
- **set_pitch**: Controls forward and backward tilt (longitudinal movement).
- **set_roll**: Controls lateral tilt (left/right movement).

In Webots, these functions are implemented so that the simulated drone responds to the same movement commands as the real drone, with some adjustments to fit the simulation environment.

## Implementation in Webots

The mentioned functions have been implemented in Webots to ensure that the simulated drone responds to the same movement commands as the physical drone controlled via `tellopy`. These functions are implemented as follows:

- **set_throttle**: Adjusts the drone’s vertical movement by modifying the `height_desired` variable based on time.
- **set_yaw**: Controls horizontal rotation, assigning the adjusted value to `yaw_desired`.
- **set_pitch**: Controls forward or backward movement by modifying `forward_desired`.
- **set_roll**: Controls lateral movement by adjusting the `sideways_desired` variable.

These functions use the `fix_range()` function to ensure that the input values are within the allowed range of -1.0 to 1.0.

## Important Notes

- The Webots API is not intended to match the exact speeds of the real drone, but to precisely replicate the **control** of the drone.
- The Webots implementation includes some differences required to adapt to the simulation environment, such as continuous updating of the vertical position based on time.

## Requirements

- [Webots](https://cyberbotics.com/) must be installed to run the simulation.
- A basic understanding of [Tellopy](https://github.com/hanyazou/TelloPy) and drone simulations in Webots is recommended.

## Usage

1. Clone this repository to your local machine.
2. Ensure that Webots is properly set up.
3. Run the simulation file in Webots, where the drone control functions will be loaded.
