# Change_My_Vote

This is the Reflex application for my master thesis called "Deep Fakes for Paper Vote Privacy Defence". The main project can be found here: https://github.com/anettehabanen/Deep_Fakes_for_Paper_Vote_Privacy_Defence

### Clone the repository:
```
git clone https://github.com/anettehabanen/Change_My_Vote
cd Change_My_Vote
```

### Python versions

1. If you have Python version 3.11 or lower

```
python3 -m venv reflex_venv   # Create an environment
source reflex_venv/bin/activate   # Activate the environment
pip install -r requirements_3-11.txt   # Install the requirements
```

2. If you have Python version 3.12 or higher

```
python3 -m venv reflex_venv   # Create an environment
source reflex_venv/bin/activate   # Activate the environment
pip install -r requirements_3-12.txt   # Install the requirements
```

To check the Python version:
```
python --version
```

### Run the project:
```
reflex run
```

The webpage can be found at http://localhost:3001/

## Example videos
The folder 'example_videos' has some images and videos of ballots and digits. To test the Reflex app, use these files when asked to upload the image of 3-digit code and a video of a voting ballot. Testing with your own videos/images is also possible.
