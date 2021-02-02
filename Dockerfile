FROM meadml/cuda10.1-cudnn7-devel-ubuntu18.04-python3.6

WORKDIR /app

RUN apt update \
    && apt update && apt install -y libsm6 libxext6 \
    && apt-get install -y libxrender-dev

RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

COPY . /app

RUN python3 -m pip install -r requirements.txt

#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8756"]
#CMD python3 tmp.py
CMD bash run.sh
#CMD ["python3", "api.py"]
