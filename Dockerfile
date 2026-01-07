# chose the base image for the container - start my image from a machine that already has Python installed
FROM python:3.12

# inside the container, the working directory is /app - current folder inside the image is /app
WORKDIR /app

# runs a shell command inside the image - OS-level stuff Matplotlib needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
  && rm -rf /var/lib/apt/lists/*

# copy the requirements.txt file from your local machine to the image
COPY requirements.txt .

# install the Python packages listed in requirements.txt into the image’s Python environment (--no-cache-dir avoids caching to reduce image size)
RUN pip install --no-cache-dir -r requirements.txt

# copy  whole project folder into /app (except files listed in .dockerignore)
COPY . .

EXPOSE 8000
# default command that will run when you start a container from this image - does not run during the build process - saved as config in the image
# when "docker run" -> FastAPI app is listening on port 8000 inside the container
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# build the image:
# docker build -t diabetes-ml-service .

# run the container that runs an image instance [-p 8000:8000 maps: laptop localhost:8000 → container :8000]: 
# docker run -p 8000:8000 diabetes-ml-service.

# it can be cleaned up when stopped (--rm):
# docker run --rm -p 8000:8000 diabetes-ml-service
# http://localhost:8000/docs
