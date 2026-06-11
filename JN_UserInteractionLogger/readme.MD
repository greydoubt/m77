# Jupyter Notebook Extension: User Interaction Logger

Jupyter Notebook Extension to track user interaXtion. 

## Set up

### Requirements

-   Ensure Jupyter Notebook is installed and can be launched from the terminal or command prompt. 
    To check this, execute ```jupyter --paths``` from either your terminal or command prompt. If the command is not found, add
    jupyter to your PATH. 

-   Ensure that *jupyter notebook extensions manager* is installed.
    If not, follow the [directions](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html) and install it. 

-   Ensure that there is a jupyter notebook configuration file in your file system. When you execute ```jupyter --paths``` in the 
    command line, you should see a list of possible directories which can contain the config file. Check to see if you can find
    ```jupyter_notebook_config.json``` AND/OR ```jupyter_notebook_config.py``` in those directories. If you cannot, then execute: ```jupyter notebook --generate-config```. Some versions of Jupyter Notebook will include the python file and not the JSON config file. The py file can be configured
    for our purposes in this project. 

### "Installing" extensions

-   Using the ```jupyter --paths``` command as a guide, identify the directory containing ```nbextensions```. If none of the paths listed contain nbextensions, 
    you might have to perform a machine level search for the directory. You might find multiple directories under the same name. The directory we want will have a
    list of subdirectories of Jupyter Notebook extensions installed through the nbextentions command we executed above. Add the CaptureLogs folder from this
    repo under ```nbextensions```. You will know that we have chosen the right directory when 'CaptureLogs' appears as one of the extensions to enable as part of
    the nbextensions tab in the Jupyter Notebook home page. 

-   Append content from config/jupyter_notebook_config.json to ```jupyter_notebook_config.json```. If this file is not present in your config directory,
    then, uncomment the commented lines in config/jupyter_notebook_config.py before appending those lines to ```jupyter_notebook_config.py```. If the json
    file does exist, only append the uncommented lines to ```jupyter_notebook_config.py```. You should find a py file in one of the config paths listed by jupyter
    --paths command. 

-   Using the ```jupyter --paths``` command as a guide, identify the directory containing jupyter libraries. Add the SaveLogs folder from this repo under 
    the site-packages directory with other jupyter libraries. If you have difficulty identifying this directory, try running jupyter notebook from 
    the command line. If you find a 'SaveLogs module not found' error, that stack trace will identify the default path for the jupyter notebook libraries. 
    Make sure to add SaveLogs directly under site-packages in that path. 
    
-   You will need to execute the follow: `jupyter contrib nbextension install --user` if you are install Jupyter Notebook extensions manager for the first time. 

-   Launch Jupyter Notebook from your command line. In the file explorer view, you should see an 'nbextensions' tab. Select the tab, and
    find 'Capture Logs' extension among the list of other available extensions. Select Capture Logs and enable the extension. Restart your
    Jupyter Notebook. 

### Checking the installation

-   Launch jupyter notebook from the command line. You shouldn't see any errors or exceptions regarding SaveLogs in the command line. 
-   Create or edit notebooks as you normally would. 
-   Open your browser's developer tools and view the console. You should see logs labeled with an '[evt]' tag. This is the data being logged
    by our extension.
-   You should also see a '[notebook name]'_log.json file in the same directory as your notebook. This file is the archive of all your user interaction. 
