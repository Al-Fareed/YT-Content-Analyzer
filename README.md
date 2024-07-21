### YT-Content-Analyzer

This is a backend application when provided with YouTube url and query related to the content present in YT-url analyzes the whole YT-video.

This application has been integrated with:
-- langchain - for ease integration with preferred AI model
-- FAISS - Vector DB to store the content

built using:
FastAPI for building API using python
poetry to manage virtual environment and to maintain dependecies

## How to use:
! Make sure you have poetry installed in your system
1. clone the repo
2. install all the dependencies using cmd ```poetry install```
3. add .env file following the template in .env.template
4. switch to poetry virtual environment ```poetry shell``` 
or
get the path of venv ```poetry env info``` and run ```emulate bash -c 'PATH_OF_VENV'```
5. Run the application ```poetry run start```