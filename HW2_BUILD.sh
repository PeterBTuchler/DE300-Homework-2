docker build -t hw2:0.1 .

# docker run -it hw2:0.1 /bin/bash

docker run -v "C:\Users\pbtuc\Documents\DE300\Homework2:/tmp/data" -it hw2:0.1 /bin/bash