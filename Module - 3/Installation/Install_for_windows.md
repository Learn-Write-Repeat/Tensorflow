<h1 align="center">TensorFlow <img src = "https://user-images.githubusercontent.com/85128689/129054838-59694bd7-cad5-4c76-82ca-b0641b814a56.png"height = "30px" width = "30px"/> Installation in Windows 
<img src =  "https://user-images.githubusercontent.com/85128689/129054168-bac83fba-bd5d-4788-b2dd-4b6b94f8b51a.png" height = "30px" width = "30px"/>
 </h1>
 
 # Versions
 - [TensorFlow](https://www.tensorflow.org/) provides two versions for windows users:
     - CPU support
     - GPU support
 - **CPU support** is for the users who system doesn't run on NVIDIA GPU.
 - **GPU support** version of TensorFlow can be used if your system has NVIDIA GPU. This version helps you in faster computation.
 
 # System requirements

- Python 3.6–3.9
    - Python 3.9 support requires TensorFlow 2.5 or later.
    - Python 3.8 support requires TensorFlow 2.2 or later.
- Windows 7 or later (64-bit)
    - Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019.
    - GPU support requires a CUDA®-enabled card.
 
 # Procedure
 - Install [Anaconda](https://www.anaconda.com/products/individual) (Wacth this [video](https://www.youtube.com/watch?v=Q-iC4VaW8ZA) to Install Anaconda)
 - Create a **.yml** file to install dependencies:
     - Locate the path of Anaconda in your system.
         - Open Anaconda Prompt and type `where anaconda`. This will show you the path of Anaconda.
     - Set a working directory to Anaconda.
     - Create a yml file.
         - `echo.>hello-tf.yml`  This creates you the **.yml** file.
     - Edit it.
         - Use Notepad i.e., use `notepad hello-tf.yml`
         - Enter following into the file
         - name: hello-tfdependencies:
             - `python=3.9` (Version of Python)
             - `jupyter`
             -  `ipython`
             -  `pandas` 
          - By the previous step you successfully prepared conda environment.  
     - Compile yml file.
         - `conda env create -f hello-tf.yml` (Make sure you've atleast 1.5 to 2gb disk space and this process takes bit time).
     - Activate Anaconda environment here.
         - `conda env list` 
         - `activate hello-tf`
- Install [TensorFlow](https://www.tensorflow.org/)
    - Using [pip](https://pypi.org/project/tensorflow/) 
        - `pip install tensorflow` 
        
- That's it TensorFlow will be successfully installed.
  
# Source
 https://www.tensorflow.org/
 
# Contributed by


<table>
    <tr>
     <td allign='centre'><a href = "https://github.com/DurgaSai-16"><img src = "https://user-images.githubusercontent.com/85128689/128643449-d2d1499b-71be-4ed0-9dfb-c5151ecc3e3a.png" height = "100px" width = "100px"/><br/><sub><b>Durga Sai Nallani</b></sub></a></td>
