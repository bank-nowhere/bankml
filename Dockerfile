FROM python:3.13

# Location of source code
ENV PROJECT_ROOT /opt/bankml
RUN mkdir -p $PROJECT_ROOT
WORKDIR $PROJECT_ROOT

# Copying dependencies
COPY ./requirements.txt .
RUN pip install setuptools wheel
RUN pip install -r requirements.txt

# Copying source code (This creates the image layers)
COPY ./src ./src
COPY ./config ./config
COPY ./examples ./examples
COPY ./notebooks ./notebooks
COPY ./data ./data

# Update and install utilities
RUN pip install --upgrade pip
RUN apt-get update -y && \
    apt-get install -y vim curl && \
    rm -rf /var/lib/apt/lists/*