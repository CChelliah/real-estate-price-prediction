FROM alpine:latest
RUN apk add --no-cache py-pip 
RUN pip3 install --upgrade pip
WORKDIR /
COPY . .
RUN pip3 --no-cache-dir install -r requirements.txt    
RUN pip install -U scikit-learn                                                                      
EXPOSE 4005
ENTRYPOINT  ["python3"]
CMD ["app.py","--host", "0.0.0.0"]