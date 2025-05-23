FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update \
    && apt-get install -y \
    cmake \
    g++ \
    git \
    libatlas-base-dev \
    libhdf5-serial-dev \
    libgl1-mesa-glx \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    swig \
    pkg-config \
    patch \
    ca-certificates \
    curl \
    gnupg \
    && ln -sf /usr/bin/swig4.0 /usr/bin/swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 (using the official script which has better error handling)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && node --version \
    && npm --version

# Verify CMake installation
RUN cmake --version

ENV AMICI_CXXFLAGS=-fopenmp
ENV AMICI_LDFLAGS=-fopenmp

# Upgrade pip and install necessary Python packages
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools>=61.2
RUN pip3 install --no-cache-dir cmake
RUN pip3 install --no-cache-dir dlib -vvv
RUN pip3 install --no-cache-dir python-libsbml>=5.17.0

RUN pip3 install --no-cache-dir h5py
RUN pip3 install --no-cache-dir pypesto[amici,petab,fides]
RUN pip3 install --no-cache-dir jupyter
RUN pip3 install --no-cache-dir pyswarm
RUN pip3 install --no-cache-dir SBMLDiagrams
RUN pip3 install --no-cache-dir streamlit
RUN pip3 install --no-cache-dir plotly

# Copy the patch file
COPY docker/patches/sbml_import_changes.patch /mnt/tmp/sbml_import_changes.patch
RUN patch /usr/local/lib/python3.10/dist-packages/amici/sbml_import.py mnt/tmp/sbml_import_changes.patch

COPY docker/patches/petab_import_changes.patch /mnt/tmp/petab_import_changes.patch
RUN patch /usr/local/lib/python3.10/dist-packages/petab/v1/problem.py mnt/tmp/petab_import_changes.patch

COPY docker/patches/pypesto_import_changes.patch /mnt/tmp/pypesto_import_changes.patch
RUN patch /usr/local/lib/python3.10/dist-packages/pypesto/petab/importer.py mnt/tmp/pypesto_import_changes.patch

COPY docker/patches/sampling_bounds.patch /mnt/tmp/sampling_bounds_changes.patch
RUN patch /usr/local/lib/python3.10/dist-packages/pypesto/visualize/sampling.py mnt/tmp/sampling_bounds_changes.patch

# Generate Jupyter Notebook configuration file
RUN jupyter notebook --generate-config

# Disable password and token authentication
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.port = 8888" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Set the working directory
WORKDIR /mnt/PolyPESTO
# Command to run when the container starts
# CMD ["jupyter", "notebook"]