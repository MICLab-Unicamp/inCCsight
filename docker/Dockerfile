FROM python:3.7.3


WORKDIR /inCCsight

COPY requirements.txt .
COPY ccprocess.py .
COPY constants.py .
COPY inputfuncs.py .
COPY parcelfuncs.py .
COPY segmfuncs.py .

COPY ./assets ./assets
COPY ./data ./data
COPY ./libcc ./libcc
COPY ./siamxt-master ./siamxt-master

EXPOSE 8080/tcp

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN cd ./siamxt-master && \
	python setup.py install && \
	cd ..

ADD app.py .

ENTRYPOINT ["python","./app.py"]
