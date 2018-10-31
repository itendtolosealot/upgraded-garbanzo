CPPFLAGS=-g

deep-learning: utils.o DeepLearning.o main.o
	nvcc utils.o DeepLearning.o main.o -o deep-learning -lcublas -lcudnn
	rm *.o

utils.o: utils.cpp
	nvcc -c $(CPPFLAGS) utils.cpp -o utils.o

DeepLearning.o: DeepLearning.cu
	nvcc -c $(CPPFLAGS) DeepLearning.cu -o DeepLearning.o

main.o: main.cpp
	nvcc -c $(CPPFLAGS) main.cpp -o main.o

clean:
	rm -fr DeepLearning $(OBJS)
