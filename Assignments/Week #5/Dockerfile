FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv 

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock" ,"model1.bin", "dv.bin", "flask_app.py", "./"]

RUN pipenv install --system --deploy

EXPOSE 9696

# As I have used waitress instead of gunicorn
ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:9696", "flask_app:app"] 

# To use Gunicorn (instead of waitress) on linux machines and 
# in the folder where project is stored, do the following:
## pipenv shell 
## pipenv install gunicorn

# Uninstall waitress by using 
## pipenv uninstall waitress  
# And check if the package waitress is Pipfile package list 

## Replace ENTRYPOINT [ "waitress-serve", "--listen=0.0.0.0:9696", "flask_app:app"] with
## ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "flask_app:app"] and build the docker image

