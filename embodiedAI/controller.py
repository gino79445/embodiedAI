import numpy as np
class controller:

    def __init__(self, env):
        self.env = env
    def stop(self):
        actions = np.zeros(12)
        while self.env._simulation_app.is_running():
            self.env.step(actions)
        return True
    def forward_backward(self, distance, velocity=0.1):
        actions = np.zeros(12)
        actions[0] = distance
        if distance < 0:
            while self.env._simulation_app.is_running():
                distance += velocity
                if distance >= 0:
                    actions[0] = 0
                    return True
                self.env.step(actions)

        else:
            while self.env._simulation_app.is_running():
                distance -= velocity
                if distance <= 0:
                    actions[0] = 0
                    return True
                self.env.step(actions)
    
    def rotation(self, angle, velocity=0.1):
        actions = np.zeros(12)
        actions[2] = angle
        if angle < 0:
            while self.env._simulation_app.is_running():
                angle += velocity
                if angle >= 0:
                    actions[2] = 0
                    return True
                self.env.step(actions)
        else:
            while self.env._simulation_app.is_running():
                angle -= velocity
                if angle <= 0:
                    actions[2] = 0
                    return True
                self.env.step(actions)

    def gripper(self, distance, velocity=0.1):

        actions = np.zeros(12)
        actions[10] = distance
        if distance < 0:
            while self.env._simulation_app.is_running():
                distance += velocity
                if distance >= 0:
                    actions[10] = 0
                    return True
                self.env.step(actions)
        else :
            while self.env._simulation_app.is_running():
                distance -= velocity
                if distance <= 0:
                    actions[10] = 0
                    return True
                self.env.step(actions)

    def arm(self, angle, velocity=0.1):
        num_actions = 12
        actions = np.zeros(num_actions)
        angle = angle.astype(np.float32)
        while self.env._simulation_app.is_running():
            # Update actions[3:10] based on distance and velocity
            actions[3:10][angle < 0] = angle[angle < 0] + velocity
            actions[3:10][angle >= 0] = angle[angle >= 0] - velocity
            # Check if any element in distance has crossed zero
            crossed_zero = np.abs(angle) < velocity

            # Update actions[3:10] for elements that have crossed zero
            actions[3:10][crossed_zero] = 0

            # Update distance for elements that haven't crossed zero
            angle[~crossed_zero] = actions[3:10][~crossed_zero]
            # Break the loop if all elements have crossed zero
            if np.all(crossed_zero):
                break
            
            # Step in the environment
            self.env.step(actions)

        return True
