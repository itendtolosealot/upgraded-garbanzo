CPPFLAGS=-rdynamic -g
GENCODE=-gencode arch=compute_30,code=sm_30

deep-learning: utils.o DeepLearning.o main.o
	nvcc -Xcompiler $(CPPFLAGS) -G  -ccbin g++ -m64 $(GENCODE) -o deep-learning utils.o DeepLearning.o main.o -I/usr/local/cuda/include -I/opt/intel/compilers_and_libraries/linux/mkl/include -lcudart -lcublas -lcudnn -L/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64_lin -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -ldl -lpthread -lm
	rm *.o

utils.o: utils.cpp
	g++ -g -I/usr/local/cuda/include -I/opt/intel/compilers_and_libraries/linux/mkl/include -c $(CPPFLAGS) utils.cpp -o utils.o

DeepLearning.o: DeepLearning.cu
	nvcc -Xcompiler $(CPPFLAGS) -G  -ccbin g++ -I/usr/local/cuda/include -I/opt/intel/compilers_and_libraries/linux/mkl/include -m64 $(GENCODE) -o DeepLearning.o -c DeepLearning.cu

main.o: main.cpp
	g++ -g -I/usr/local/cuda/include -I/opt/intel/compilers_and_libraries/linux/mkl/include  -c $(CPPFLAGS) main.cpp -o main.o

clean:
	rm -rf deep-learning $(OBJS)
