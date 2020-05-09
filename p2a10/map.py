import os
import time
import random
import numpy as np
import math
from PIL import Image as PILImage
import cv2
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture

from ai_t3d import DNN,ReplayBuffer,TD3

Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429') #1074
Config.set('graphics', 'height', '660') #760

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_obs(img, x, y, gx, gy, model, car):
  x, y = (int(x), int(y))
  # rotated_image = np.rot90(img)
  crop_img = img[x-25:x+25, y-25:y+25]
  # cv2.imshow("Crop", crop_img)
  # cv2.waitKey(1)
  crop_img = crop_img / 255
  crop_img = PILImage.fromarray(crop_img)
  obs = model(crop_img)
  xx = gx - x
  yy = gy - y
  orientation = Vector(*car.velocity).angle((xx,yy))/180.
  distance = math.sqrt((x - gx)**2 + (y - gy)**2)
  obs = np.append(obs, orientation)
  obs = np.append(obs, -orientation)
  obs = np.append(obs, math.tanh(distance))
  return obs

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = int(rotation)
        self.angle = int(self.angle + self.rotation)

class Ball_goal(Widget):
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
goal_x = 580
goal_y = 340
last_x = 580
last_y = 340
last_reward = 0
avg_done_reward = 24

class Game(Widget):
    car = ObjectProperty(None)
    ballg = ObjectProperty(None)
    #Setup
    start_timesteps = 3e3 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 1e3 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 3e5 # Total number of iterations/timesteps
    save_models = False # Boolean checker whether or not to save the pre-trained model
    expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 100 # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
    
    
    img = cv2.imread("./images/mask.png") #plain.png
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    state_dim = 13
    action_dim = 1
    max_action = 25.0

    policy = TD3(state_dim, action_dim, max_action)
    env_state = DNN()
    # env_state.load_state_dict(torch.load('./pytorch_models/cnn.pth'))
    replay_buffer = ReplayBuffer()

    max_episode_steps = 1000
    total_timesteps = 0
    episode_reward = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    living_penalty = 0
    last_distance = 0
    road_travelled = 0
    obs = []
    file_name = "T3D_car_1"
    policy.load(file_name, './pytorch_models/')

    if save_models and not os.path.exists("./pytorch_models"):
      os.makedirs("./pytorch_models")

    def serve_car(self):
        # print(self.center)
        self.car.center = [865 , 640]
        self.car.velocity = Vector(1, 0)

    def update(self, dt):

        global last_reward
        global last_x
        global last_y
        global goal_x
        global goal_y
        global avg_done_reward

        if self.total_timesteps > self.max_timesteps:
          CarApp().stop()


        if self.done:
          if self.total_timesteps != 0:
            print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
            self.policy.train(self.replay_buffer, self.episode_timesteps, self.batch_size, self.discount, self.tau, self.policy_noise, self.noise_clip, self.policy_freq)

          if self.timesteps_since_eval >= self.eval_freq:
            self.timesteps_since_eval %= self.eval_freq
            self.policy.save(self.file_name, directory="./pytorch_models")
            torch.save(self.env_state.state_dict(), './pytorch_models/cnn.pth')
            print("Model Saved...")
          
          self.car.x = random.randint(int(self.width*0.3), int(self.width*0.7))
          self.car.y = random.randint(int(self.height*0.3), int(self.height*0.7))
          while self.img[int(math.floor(self.car.x)),int(math.floor(self.car.y))] == 255:
            self.car.x = random.randint(int(self.width*0.2), int(self.width*0.8))
            self.car.y = random.randint(int(self.height*0.2), int(self.height*0.8))
          print("Reinit", self.car.x, self.car.y, self.img[int(self.car.x),int(self.car.y)])
          self.obs = get_obs(self.img, self.car.x, self.car.y, goal_x, goal_y, self.env_state, self.car)
          self.done = False
          self.episode_reward = 0
          self.episode_timesteps = 0
          self.episode_num += 1
          last_reward = 0
          last_x = 580
          last_y = 340
          self.living_penalty = 0
          self.road_travelled = 0
        
        if self.total_timesteps < self.start_timesteps:
          self.action = np.array([random.uniform(-1, 1)*25])
        else: # After 10000 timesteps, we switch to the model
          self.action = self.policy.select_action(np.array(self.obs))
          # print(self.action)

          if self.expl_noise != 0:
            self.action = (self.action + np.random.normal(0, self.expl_noise, size=1)).clip(-25, 25)
        
        self.car.move(int(self.action[0]))
        new_obs = get_obs(self.img, self.car.x, self.car.y, goal_x, goal_y, self.env_state, self.car)
        distance = abs(np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2))
        prev_dist = abs(np.sqrt((self.car.x - last_x)**2 + (self.car.y - last_y)**2))
        # print(prev_dist)
        self.car.velocity = Vector(1.0, 0).rotate(self.car.angle)

        #Reward Calculation
        self.living_penalty += 0.02
        if self.img[last_x,last_y] == 0 and self.img[int(math.floor(self.car.x)),int(math.floor(self.car.y))] == 255:
          self.living_penalty *= 1.01
          self.living_penalty += 0.1
        elif self.img[last_x,last_y] == 255 and self.img[int(math.floor(self.car.x)),int(math.floor(self.car.y))] == 255:
          self.living_penalty += 0.15
          self.road_travelled += -0.05
        elif self.img[last_x,last_y] == 0 and self.img[int(math.floor(self.car.x)),int(math.floor(self.car.y))] == 0:
          self.living_penalty += -0.005
          self.road_travelled += 0.1
          self.living_penalty *= 0.999
        else:
          self.living_penalty *= 1.01
          self.living_penalty += 0.1

        if prev_dist<0.5:
          self.living_penalty += 0.05
        elif prev_dist>0.5 and prev_dist<1.5:
          self.living_penalty += 0.005
        else:
          self.living_penalty -= 0.01

        if self.img[int(math.floor(self.car.x)),int(math.floor(self.car.y))] == 255:
            if distance<100:
              self.living_penalty *= 1.01
            if distance > self.last_distance:
                last_reward += -self.living_penalty*1.6
            else:
                last_reward += -self.living_penalty*1.4
        else: # otherwise
            if distance<100:
              self.living_penalty *= 0.99
            if distance < self.last_distance:
                last_reward += -self.living_penalty
            else:
                last_reward += -self.living_penalty*1.2

        # last_reward += -distance/120

        if self.car.x <= 28:
            self.car.x = 28
            last_reward += -355 -self.living_penalty -distance/40
            self.done = True
        if self.car.x >= self.width - 28:
            self.car.x = self.width - 28
            last_reward += -355 -self.living_penalty -distance/40
            self.done = True
        if self.car.y <= 28:
            self.car.y = 28
            last_reward += -355 -self.living_penalty -distance/40
            self.done = True
        if self.car.y >= self.height - 28:
            self.car.y = self.height - 28
            last_reward += -355 -self.living_penalty -distance/40
            self.done = True

        if distance < 4:
            goal_x = random.randint(int(self.width*0.35), int(self.width*0.65))
            goal_y = random.randint(int(self.height*0.35), int(self.height*0.65))
            while self.img[goal_x,goal_y] == 255:
              goal_x = random.randint(int(self.width*0.2), int(self.width*0.8))
              goal_y = random.randint(int(self.height*0.2), int(self.height*0.8))
            last_reward += 15 - self.living_penalty
            self.living_penalty = 0
            self.done = True

        if self.episode_timesteps>750: #self.living_penalty > 15 and self.total_timesteps>200:
          if self.living_penalty > 15:
            self.done = True
          # last_reward += -distance/80

        # if self.img[last_x,last_y] == 255 and self.img[int(math.floor(self.car.x)),int(math.floor(self.car.y))] == 255 and self.total_timesteps>self.start_timesteps:
        #   self.done = True

        done_bool = float(self.done)
        
        if self.done == True:
          last_reward += self.road_travelled
          print("reset")
          if self.img[int(math.floor(self.car.x)),int(math.floor(self.car.y))] == 255:
            last_reward -= 8 -distance/40
          else:
            last_reward += 8 -distance/40

        self.replay_buffer.add((self.obs, new_obs, self.action, last_reward, done_bool))
        print(self.total_timesteps, self.action, last_reward, self.living_penalty, self.img[int(self.car.x),int(self.car.y)])
        print(goal_x, goal_y, self.car.x, self.car.y, done_bool, distance)
        last_reward = 0

        self.last_distance = distance
        last_x = int(math.floor(self.car.x))
        last_y = int(math.floor(self.car.y))
        self.ballg.pos = (goal_x, goal_y)
        self.obs = new_obs
        self.episode_timesteps += 1
        self.total_timesteps += 1
        self.timesteps_since_eval += 1
        self.episode_reward += last_reward * 0.01

        if self.save_models: 
          self.policy.save("%s" % (self.file_name), directory="./pytorch_models")
          torch.save(self.env_state.state_dict(), './pytorch_models/cnn.pth')

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent


# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
    # cv2.destroyAllWindows()