CC = g++
FLAGS = -Wall -g -lOpenCL -I . -D CL_TARGET_OPENCL_VERSION=300 -D CL_HPP_TARGET_OPENCL_VERSION=300

all:
	$(CC) -c OpenCLstuff.cpp $(FLAGS)
	# $(CC) -c buffer.cpp $(FLAGS)
	$(CC) DataSet.cpp $(FLAGS)	


main:
	$(CC) main.cpp

clean: 
	rm -rf *.o
