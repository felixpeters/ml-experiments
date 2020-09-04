FROM python

COPY requirements.txt /ml-env/
WORKDIR /ml-env
RUN pip install -r requirements.txt
