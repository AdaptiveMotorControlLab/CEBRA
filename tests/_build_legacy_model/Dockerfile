FROM python:3.12-slim AS base
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

FROM base AS cebra-0.4.0-scikit-learn-1.4
RUN pip install cebra==0.4.0 "scikit-learn<1.5"
WORKDIR /app
COPY create_model.py .
RUN python create_model.py

FROM base AS cebra-0.4.0-scikit-learn-1.6
RUN pip install cebra==0.4.0 "scikit-learn>=1.6"
WORKDIR /app
COPY create_model.py .
RUN python create_model.py

FROM base AS cebra-rc-scikit-learn-1.4
# NOTE(stes): Commit where new scikit-learn tag logic was added to the CEBRA class.
# https://github.com/AdaptiveMotorControlLab/CEBRA/commit/5f46c3257952a08dfa9f9e1b149a85f7f12c1053
RUN pip install git+https://github.com/AdaptiveMotorControlLab/CEBRA.git@5f46c3257952a08dfa9f9e1b149a85f7f12c1053 "scikit-learn<1.5"
WORKDIR /app
COPY create_model.py .
RUN python create_model.py

FROM base AS cebra-rc-scikit-learn-1.6
# NOTE(stes): Commit where new scikit-learn tag logic was added to the CEBRA class.
# https://github.com/AdaptiveMotorControlLab/CEBRA/commit/5f46c3257952a08dfa9f9e1b149a85f7f12c1053
RUN pip install git+https://github.com/AdaptiveMotorControlLab/CEBRA.git@5f46c3257952a08dfa9f9e1b149a85f7f12c1053 "scikit-learn>=1.6"
WORKDIR /app
COPY create_model.py .
RUN python create_model.py

FROM scratch
COPY --from=cebra-0.4.0-scikit-learn-1.4 /app/cebra_model.pt /cebra_model_cebra-0.4.0-scikit-learn-1.4.pt
COPY --from=cebra-0.4.0-scikit-learn-1.6 /app/cebra_model.pt /cebra_model_cebra-0.4.0-scikit-learn-1.6.pt
COPY --from=cebra-rc-scikit-learn-1.4 /app/cebra_model.pt /cebra_model_cebra-rc-scikit-learn-1.4.pt
COPY --from=cebra-rc-scikit-learn-1.6 /app/cebra_model.pt /cebra_model_cebra-rc-scikit-learn-1.6.pt
