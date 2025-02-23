import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
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

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def load_img(self, img):
        loader = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        img = loader(img).float()
        batch = torch.as_tensor(img).unsqueeze(0)
        return batch

    def forward(self, img):
        x = self.load_img(img)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        out = x.data.cpu().numpy()[0]
        return out

class ReplayBuffer(object):

  def __init__(self, max_size=1e5):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    # ind = np.random.randint(0, len(self.storage), size=batch_size)
    ind = random.sample(range(0, len(self.storage)), batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.layer_1 = nn.Linear(state_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, action_dim)
    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.layer_1(x))
    x = F.relu(self.layer_2(x))
    x = self.max_action * torch.tanh(self.layer_3(x)) 
    return x

class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.layer_1 = nn.Linear(state_dim + action_dim, 400)
    self.layer_2 = nn.Linear(400, 300)
    self.layer_3 = nn.Linear(300, 1)
    # Defining the second Critic neural network
    self.layer_4 = nn.Linear(state_dim + action_dim, 400)
    self.layer_5 = nn.Linear(400, 300)
    self.layer_6 = nn.Linear(300, 1)

  def forward(self, x, u):
    xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.layer_4(xu))
    x2 = F.relu(self.layer_5(x2))
    x2 = self.layer_6(x2)
    return x1, x2

  def Q1(self, x, u):
    xu = torch.cat([x, u], 1)
    x1 = F.relu(self.layer_1(xu))
    x1 = F.relu(self.layer_2(x1))
    x1 = self.layer_3(x1)
    return x1

# Building the whole Training Process into a class
class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def get_obs(img, x, y, gx, gy, model, car):
  x, y = (int(x), int(y))
  rotated_image = np.rot90(img)
  crop_img = rotated_image[y-20:y+20, x-20:x+20]
  crop_img = crop_img / 255
  crop_img = PILImage.fromarray(crop_img)
  obs = model(crop_img)
  xx = gx - x
  yy = gy - y
  orientation = Vector(*car.velocity).angle((xx,yy))/180.
  distance = np.sqrt((x - gx)**2 + (y - gy)**2)
  obs = np.append(obs, orientation)
  obs = np.append(obs, -orientation)
  # obs = np.append(obs, distance)
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

class Game(Widget):
    car = ObjectProperty(None)
    ballg = ObjectProperty(None)
    #Setup
    start_timesteps = 1e3 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 5e2 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 5e4 # Total number of iterations/timesteps
    save_models = True # Boolean checker whether or not to save the pre-trained model
    expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 100 # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
    goal_x = 120
    goal_y = 700
    img = cv2.imread("./images/plain.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    state_dim = 12
    action_dim = 1
    max_action = 1.0

    policy = TD3(state_dim, action_dim, max_action)
    env_state = DNN()
    replay_buffer = ReplayBuffer()

    max_episode_steps = 1000
    total_timesteps = 0
    episode_reward = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    t0 = time.time()
    last_reward = 0
    living_penalty = 0
    last_distance = 0
    swap = 0
    obs = []
    action = 5
    file_name = "T3D_car_0"

    if save_models and not os.path.exists("./pytorch_models"):
      os.makedirs("./pytorch_models")

    def serve_car(self):
        # print(self.center)
        self.car.center = [865 , 640]
        self.car.velocity = Vector(1, 0)

    def update(self, dt):

        # global last_reward
        # global last_x
        # global last_y
        # global last_distance
        # global goal_x
        # global goal_y
        # global swap
        # global living_penalty


        if self.total_timesteps > self.max_timesteps:
          CarApp().stop()
        print(self.total_timesteps)


        if self.done:
          if self.total_timesteps != 0:
            print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, self.episode_num, self.episode_reward))
            self.policy.train(self.replay_buffer, self.episode_timesteps, self.batch_size, self.discount, self.tau, self.policy_noise, self.noise_clip, self.policy_freq)

          if self.timesteps_since_eval >= self.eval_freq:
            self.timesteps_since_eval %= self.eval_freq
            self.policy.save(self.file_name, directory="./pytorch_models")
          
          self.car.x = random.randint(int(self.width*0.2), int(self.width*0.8))
          self.car.y = random.randint(int(self.height*0.2), int(self.height*0.8))
          self.obs = get_obs(self.img, self.car.x, self.car.y, self.goal_x, self.goal_y, self.env_state, self.car)
          self.done = False
          self.episode_reward = 0
          self.episode_timesteps = 0
          self.episode_num += 1
          self.last_reward = 0
        
        if self.total_timesteps < self.start_timesteps:
          self.action = np.array([random.uniform(-1, 1)])
        else: # After 10000 timesteps, we switch to the model
          self.action = self.policy.select_action(np.array(self.obs))

          if self.expl_noise != 0:
            self.action = (self.action + np.random.normal(0, self.expl_noise, size=1)).clip(-10, 10)*10
        
        self.car.move(int(self.action[0]*10))
        new_obs = get_obs(self.img, self.car.x, self.car.y, self.goal_x, self.goal_y, self.env_state, self.car)
        # print("obs", new_obs.shape, type(new_obs))
        distance = np.sqrt((self.car.x - self.goal_x)**2 + (self.car.y - self.goal_y)**2)
        if self.img[int(self.car.x),int(self.car.y)] > 0.8:
            self.car.velocity = Vector(0.4, 0).rotate(self.car.angle)
            # print(1, goal_x, goal_y, distance, int(car.x),int(car.y), im.read_pixel(int(car.x),int(car.y)))
            if distance > self.last_distance:
                self.last_reward = -1.8
            else:
                self.last_reward = -1.2

        elif self.img[int(self.car.x),int(self.car.y)] > 0.1 and self.img[int(self.car.x),int(self.car.y)] < 0.7:
            self.car.velocity = Vector(0.8, 0).rotate(self.car.angle)
            # print(1, goal_x, goal_y, distance, int(car.x),int(car.y), im.read_pixel(int(car.x),int(car.y)))
            if distance > self.last_distance:
                self.last_reward = -0.9
            else:
                self.last_reward = -0.6
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            # print(0, goal_x, goal_y, distance, int(car.x),int(car.y), im.read_pixel(int(car.x),int(car.y)))
            if distance < self.last_distance:
                self.last_reward = 0.3
                # living_penalty -= 0.002
            else:
                self.last_reward = -0.2
                # living_penalty += 0.002

        if self.car.x <= 25:
            self.car.x = 25
            self.last_reward = -2.7
        if self.car.x >= self.width - 25:
            self.car.x = self.width - 25
            self.last_reward = -2.7
        if self.car.y <= 25:
            self.car.y = 25
            self.last_reward = -2.7
        if self.car.y >= self.height - 25:
            self.car.y = self.height - 25
            self.last_reward = -2.7

        self.living_penalty += 0.02
        self.last_reward -= self.living_penalty
        print(self.last_reward)

        if distance < 15:
            if self.swap == 1:
                self.goal_x = 865
                self.goal_y = 660
                self.swap = 2
                self.living_penalty = 0
                self.last_reward = 0
                self.done = True
            elif self.swap == 2:
                self.goal_x = 120
                self.goal_y = 700
                self.swap = 0
                self.living_penalty = 0
                self.last_reward = 0
                self.done = True
            else:
                self.goal_x = 660
                self.goal_y = 60
                self.swap = 1
                self.living_penalty = 0
                self.last_reward = 0
                self.done = True
        self.last_distance = distance
        if self.last_reward < -12 :
          self.done = True
          self.last_reward -= 10

        self.ballg.pos = (self.goal_x, self.goal_y)
        # new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if self.episode_timesteps + 1 == 1000 else float(self.done)
        self.episode_reward += self.last_reward
        # print("action", self.action.shape, type(self.action))
        self.replay_buffer.add((self.obs, new_obs, self.action, self.last_reward, done_bool))
        print(self.goal_x, self.goal_y, self.car.x, self.car.y, done_bool)

        self.obs = new_obs
        self.episode_timesteps += 1
        self.total_timesteps += 1
        self.timesteps_since_eval += 1

        if self.save_models: self.policy.save("%s" % (self.file_name), directory="./pytorch_models")

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent


# Running the whole thing
if __name__ == '__main__':
    CarApp().run()