#!/usr/bin/env python
# coding: utf-8

# # <center>付録: matplotlibによるアニメーションのテスト</center>
# Author: 藤原 義久 <yoshi.fujiwara@gmail.com>    

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# In[2]:


# Magic command for matplotlib to work interactively 
# Call twice to avoid a problem (https://gist.github.com/shoeffner/07c1c9ba7407684141372e2e862d0503)
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'notebook')

# 単一の図の場合
fig = plt.figure(figsize=(6,4))

def update(frame):
    plt.cla()
    x = np.arange(0,10,0.1)
    dx = float(frame)*0.1
    y = np.sin(x-dx)
    plt.plot(x, y, "b")
    plt.xlabel("x", fontsize=14)
    plt.ylabel("sin(x-0.1*a)", fontsize=14)
    plt.title("a=" + str(frame))

ani = animation.FuncAnimation(fig, update, frames=range(50), interval=100)


# In[3]:


get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'notebook')

# 複数の図の場合
fig, axs = plt.subplots(2)

def update(frame):
    axs[0].cla()
    axs[1].cla()
    x = np.arange(0,10,0.1)
    dx = float(frame)*0.1
    y = np.sin(x-dx)
    z = np.cos(x-dx)
    axs[0].plot(x,y,"b")
    axs[1].plot(x,z,"r")
    axs[0].set_ylabel("sin(x-dx)", fontsize=14)
    axs[0].set_title("a=" + str(frame))
    axs[1].set_xlabel("x", fontsize=14)
    axs[1].set_ylabel("cos(x-dx)", fontsize=14)

ani = animation.FuncAnimation(fig, update, frames=range(50), interval=100)


# In[ ]:




