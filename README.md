## description

this is a project inspired by @[TeaPearce](https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/commits?author=TeaPearce) and his previous work on behavioral cloning csgo in deathmatch mode.

our purpose is to make an another approach in csgo behavioral cloning by acquiring our data parsing demos.

with that, we believe we can get cleaner data on larger scale and conquer csgo competitive , the charming pearl of all.



## 1 min startup for training your agent

- get the [dataset](https://www.kaggle.com/datasets/kissjonh/csgo-competitive-dataset) on kaggle, unzip it in you project root.
- python train.py -name ./net_20.pkl -lr 0.0011 -epoch 1000 -batch 50
  - name: the name of your model, if no such model then it creates one starting by net_0.pkl
  - lr:  because we use adamw optimizer, it's useless. change the code if you need it.
  - epoch: epoch number you want to run
  - batch: the batch size.

## 1 min startup for agent field test(run in game)

- put your model in game folder, change model path in run_agent.py
- start csgo, type sv_cheats 1; cl_draw_only_deathnotices 1; host_timescale 0.xx; in console (it depends on your pc, just make sure it runs 16 per second)
- python run_agent.py and you are ready to go.



## model description

the model assume that you are runing the forward action 16 times per second.

data: the video was 32fps per second and we keep 16 frames and abandon the rest.

input: the previous 16 frames of game video.  shape: (batch, 16, channel,  width, height)

output: [w a s d is_fire is_scope is_jump is_crouch is_walking is_reload is_e switch mouse_x mouse_y] each one is an index num.

​              **wasd  is_fire is_scope is_jump is_crouch is_walking is_reload is_e** are binary index 0/1

​			  switch 0 1 2 3 4 5

​              mouse_x mouse_y  are chosen from possible action lists.



## contact

for any problem,feel free to propose an issue.

or you can mail me here:  mengshi2022@ia.ac.cn



## credit

thanks to @TeaPearce for his innovitive idea.

thanks to @5eplay for providing game demos for academic research.

thanks to @akiver for [the demo manager]( https://github.com/akiver/CSGO-Demos-Manager) and @[pnxenopoulos](https://github.com/pnxenopoulos) for [awpy](https://github.com/pnxenopoulos/awpy) the parser





