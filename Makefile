CPPFLAGS= -rdynamic -lineinfo -g
GENCODE=-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_72,code=compute_72

deep-learning: utils.o DeepLearning.o main.o
	nvcc -Xcompiler $(CPPFLAGS) -ccbin g++ -m64 $(GENCODE) -o deep-learning utils.o DeepLearning.o main.o -I/usr/local/cuda/include -lcudart -lcublas -lcudnn -lstdc++ -lm 
	rm *.o

utils.o: utils.cpp
	g++ -g -I/usr/local/cuda/include  -c $(CPPFLAGS) utils.cpp -o utils.o

DeepLearning.o: DeepLearning.cu
	nvcc -Xcompiler $(CPPFLAGS) -ccbin g++ -I/usr/local/cuda/include  -m64 $(GENCODE) -o DeepLearning.o -c DeepLearning.cu

main.o: main.cpp
	g++ -g -I/usr/local/cuda/include  -c $(CPPFLAGS) main.cpp -o main.o

clean:
	rm -rf deep-learning $(OBJS)
