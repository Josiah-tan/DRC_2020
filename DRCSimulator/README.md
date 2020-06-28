# USRC's Drone Racing Competition Simulator

If you're a developer looking to improve this repository, please look at `developers.md` in this same directory.

## Running the simulator
1. Open a terminal at the directory and type in `python3 simulatorServer.py`.
2. In a web browser, go to `http://localhost:8000`. You should see a car floating in mid air. (That's normal. Cars do this all the time.)
3. In a separate terminal, run `python3 car_edited.py`. You should see the following things happen:
    - The car in your browser window starts moving in a circle.
    - A window pops up, showing the car's point of view.
4. Of the two windows in the browser, you can click and drag the top one to change your view into a birds-eye view.

## Controlling the Robot

1. Use the terminal in which you ran  `python3 car_edited.py` to type in w/a/s/d and then hit enter
2. After this the robot will move with a speed specified by `speed_increment` and rotate at an angle specified by `rotation_increment`

Have fun!