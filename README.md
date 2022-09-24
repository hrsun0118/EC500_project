# EC500_project

This folder contains the paper, the presentation and all the src code that's implemented in the Mars Rover Simulation for autonomous driving. The key ML method used is Imitation Learning(IL). The IL agent is trained through a series of data collection by driving the car around the rover inside the simulation. 

Data Collection Process: 
* To traverse through the long corridor (with Martian mountain walls on the side) inside the simulation, we drove the car closer to the wall.
* To turn at the end of a long corridor, we trained the rover to turn at various terrain. Based on the color of the image taken when turning, rover is trained to turn when encountering objects blocking it's field of view.
* To avoid obstacles, we drove the rover close to different obstacles and purposely turn right/left to train it to turn. Ideally, it's best to turn in only 1 direction to avoid confusion when rover is faced with this situation. As a result, we did multiple trails of training to get the optimal result.
